import numpy as np
import taichi as ti
import taichi.math as tm

from sampling import build_coordinate_system
from trace import trace_inner, trace_outer
from shading import interpolate_normal

# --- Table 2 (paper), mm^-1, at R=700nm G=546.1nm B=435.8nm ---
SIGMA_A_EUMELANIN   = np.array([22.150,   50.632,  107.330], dtype=np.float32)
SIGMA_A_PHEOMELANIN = np.array([ 8.875,   28.864,   84.291], dtype=np.float32)
SIGMA_A_OXYHB       = np.array([ 0.1553,  26.704,   71.123], dtype=np.float32)
SIGMA_A_DEOXYHB     = np.array([ 0.9608,  27.453,  292.932], dtype=np.float32)
SIGMA_A_BILIRUBIN   = np.array([ 0.00026,  0.00017,  0.1268], dtype=np.float32)
SIGMA_A_OTHER       = np.array([ 0.02663,  0.0472,   0.1452], dtype=np.float32)

# reduced scattering mu_s', also Table 2
MU_S_PRIME_EPI  = np.array([4.6483, 6.2014, 8.0584], dtype=np.float32)
MU_S_PRIME_DERM = np.array([2.9329, 4.0421, 5.4101], dtype=np.float32)

BETA = 0.75    # oxygenated fraction of hemoglobin; paper fixes this
G = 0.0        # HG asymmetry

# [0] escaped
# [1] killed by roulette (unbiased, not a loss)
# [2] lost the geometry (both traces missed)
# [3] hit the hard step cap (real energy loss)
walk_stats = ti.field(dtype=ti.i32, shape=4)

# summed exit throughput per channel; divide by walk_stats[0] for the mean.
# f64, NOT f32: this runs to ~1e8 additions of numbers around 0.3, and f32
# has only ~7 digits, so the running sum stops registering them and the mean
# silently drifts downward as the render goes on.
walk_rgb = ti.field(dtype=ti.f64, shape=3)

# how many times a walk was turned back at the outer surface by total
# internal reflection instead of escaping
walk_tir = ti.field(dtype=ti.i32, shape=())

# diagnostics: [0] scattering events, [1] wall crossings, [2] steps spent in
# the epidermis, [3] total steps. f64 because these run to ~1e11.
walk_diag = ti.field(dtype=ti.f64, shape=4)

# ceiling on throughput after a pdf division. slight bias, but an unlucky
# sample can otherwise return an enormous value and produce a firefly pixel
MAX_THROUGHPUT = 4.0

# roulette only starts after this many steps, so short walks run untouched
ROULETTE_START = 512


def epidermis_absorption(v, alpha):
    """Eq. 4.  v = melanin fraction, alpha = eumelanin proportion"""
    melanin = alpha * SIGMA_A_EUMELANIN + (1.0 - alpha) * SIGMA_A_PHEOMELANIN
    return v * melanin + (1.0 - v) * SIGMA_A_OTHER


def dermis_absorption(tau):
    """Eq. 6.  tau = hemoglobin fraction"""
    hb = BETA * SIGMA_A_OXYHB + (1.0 - BETA) * SIGMA_A_DEOXYHB + SIGMA_A_BILIRUBIN
    return tau * hb + (1.0 - tau) * SIGMA_A_OTHER


def to_sigma_s(mu_s_prime, g=G):
    """Table 2 already gives mu_s'. With g = 0 the reduced coefficient IS the
    scattering coefficient, so it is used directly. (The 1/(1-g) conversion
    made the mean free path 5x shorter and starved the walk of reach.)"""
    return mu_s_prime


# --- biological parameters (Table 3 ranges) ---
MELANIN_FRACTION = 0.005
MELANIN_BLEND    = 0.5
HEMOGLOBIN_FRAC  = 0.02


sigma_a_epi_np = epidermis_absorption(MELANIN_FRACTION, MELANIN_BLEND)
sigma_s_epi_np = to_sigma_s(MU_S_PRIME_EPI)
sigma_t_epi_np = sigma_a_epi_np + sigma_s_epi_np

sigma_a_derm_np = dermis_absorption(HEMOGLOBIN_FRAC)
sigma_s_derm_np = to_sigma_s(MU_S_PRIME_DERM)
sigma_t_derm_np = sigma_a_derm_np + sigma_s_derm_np

SIGMA_A_EPI = tm.vec3(*sigma_a_epi_np)
SIGMA_S_EPI = tm.vec3(*sigma_s_epi_np)
SIGMA_T_EPI = tm.vec3(*sigma_t_epi_np)

SIGMA_A_DERM = tm.vec3(*sigma_a_derm_np)
SIGMA_S_DERM = tm.vec3(*sigma_s_derm_np)
SIGMA_T_DERM = tm.vec3(*sigma_t_derm_np)

INV_4PI = 1.0 / (4.0 * tm.pi)


@ti.func
def hg_phase(cos_theta, g):
    d = 1.0 + g * g - 2.0 * g * cos_theta
    return INV_4PI * (1.0 - g * g) / (d * tm.sqrt(tm.max(d, 1e-8)))


@ti.func
def sample_hg(wi, g):
    """New direction after a scattering event. The frame is built around the
    direction of travel, not a surface normal - there is no surface in here."""
    u0 = ti.random(ti.f32)
    u1 = ti.random(ti.f32)

    cos_theta = 0.0
    if abs(g) < 1e-3:
        cos_theta = 1.0 - 2.0 * u0                       # isotropic
    else:
        s = (1.0 - g * g) / (1.0 + g - 2.0 * g * u0)
        cos_theta = (1.0 + g * g - s * s) / (2.0 * g)

    cos_theta = tm.clamp(cos_theta, -1.0, 1.0)
    sin_theta = tm.sqrt(tm.max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * tm.pi * u1

    t, b = build_coordinate_system(wi)
    wo = (sin_theta * tm.cos(phi) * t
          + sin_theta * tm.sin(phi) * b
          + cos_theta * wi)

    return tm.normalize(wo), hg_phase(cos_theta, g)


# refract
IOR_SKIN = 1.4
# reflect (head on)
F0_SKIN = ((IOR_SKIN - 1.0) / (IOR_SKIN + 1.0)) ** 2   # ~0.028


@ti.func
def fresnel_dielectric_exit(cos_i, ior):
    """Reflectance for light LEAVING a medium of index `ior` into air.

    Schlick cannot be used here. Schlick is an approximation fitted for the
    air-to-dense direction, where reflectance rises smoothly from ~3% to 100%
    at grazing angles. Going the other way there is a hard cutoff: past the
    critical angle (about 46 degrees for ior 1.4) NOTHING gets out, all of it
    reflects back inside. Schlick has no cutoff and would report ~3% there.

    cos_i is measured against the OUTWARD normal, so it is >= 0.
    Returns 1.0 under total internal reflection.
    """
    F = 1.0
    # Snell: sin_t = ior * sin_i. TIR when that exceeds 1.
    sin_t2 = ior * ior * (1.0 - cos_i * cos_i)
    if sin_t2 < 1.0:
        cos_t = tm.sqrt(tm.max(0.0, 1.0 - sin_t2))
        r_s = (ior * cos_i - cos_t) / (ior * cos_i + cos_t + 1e-8)
        r_p = (ior * cos_t - cos_i) / (ior * cos_t + cos_i + 1e-8)
        F = 0.5 * (r_s * r_s + r_p * r_p)
    return F


@ti.func
def refract_into_medium(direction, normal, ior):
    cos_i = -tm.dot(direction, normal)
    eta = 1.0 / ior
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    result = direction
    if k >= 0.0:
        result = tm.normalize(eta * direction + (eta * cos_i - tm.sqrt(k)) * normal)
    return result


MAX_WALK_STEPS = 2048
SURF_EPS = 1e-3       # mm; epidermis is 0.05 mm, so this is 2% of it


@ti.func
def random_walk_sss(start_pos, start_dir, px, py):
    """
    Walk inside the skin until the light exits through the outer surface.
    start_pos must already be just below the outer surface.

    The layer is derived from geometry every step rather than tracked with a
    flag: the trace's `front` value is -1 when the inner mesh is struck from
    within, which means we are in the dermis, and +1 when struck from outside,
    which means we are in the epidermis. This comes from the geometric normal,
    so unlike a dot product against the interpolated normal it cannot flip the
    wrong way near a silhouette. It also cannot drift.

    Free-flight distances are sampled from ONE colour channel, chosen in
    proportion to how much throughput is left in each. The weights are
    normalised to sum to 1 and the mixture pdf uses those same weights, which
    keeps the estimator unbiased.
    (Balance heuristic; cf. Wilkie et al., Hero Wavelength Spectral Sampling.)

    Reaching the outer surface is NOT the same as leaving through it. Light
    arriving from inside at a steep angle is totally internally reflected and
    has to keep bouncing until it happens to arrive within the escape cone.
    That trapped light is a large fraction of the total, and letting it all
    out makes the skin roughly twice as bright as it should be.
    """
    pos = start_pos
    dir = start_dir
    throughput = tm.vec3(1.0, 1.0, 1.0)

    escaped = 0
    exit_tri = -1
    exit_u = 0.0
    exit_v = 0.0
    lost = 0
    rouletted = 0

    n_scat = 0.0
    n_wall = 0.0
    n_epi = 0.0
    n_step = 0.0

    for step in range(MAX_WALK_STEPS):
        n_step += 1.0
        t_out, tri_out, u_out, v_out, front_out = trace_outer(pos, dir, px, py)
        t_in, tri_in, u_in, v_in, front_in = trace_inner(pos, dir, px, py)

        if t_out > 1e29 and t_in > 1e29:
            lost = 1
            break

        # --- which layer are we in? decided by geometry, not history ---
        # front_in < 0 means the inner shell is hit from the inside, so the
        # walk is currently within it: the dermis.
        in_epidermis = 1
        if tri_in >= 0 and front_in < 0.0:
            in_epidermis = 0

        if in_epidermis == 1:
            n_epi += 1.0

        sigma_t = SIGMA_T_EPI
        sigma_s = SIGMA_S_EPI
        if in_epidermis == 0:
            sigma_t = SIGMA_T_DERM
            sigma_s = SIGMA_S_DERM

        # --- choose which channel to sample the free flight from ---
        # weights are relative shares that sum to 1, so the surviving channel
        # (usually red) gets most of the samples
        w = tm.max(throughput, 0.0)
        wsum = w[0] + w[1] + w[2]

        c = 0
        if wsum > 1e-8:
            w = w / wsum
            r = ti.random(ti.f32)
            if r < w[0]:
                c = 0
            elif r < w[0] + w[1]:
                c = 1
            else:
                c = 2
        else:
            # everything absorbed; fall back to uniform so we still terminate
            c = ti.min(int(ti.random(ti.f32) * 3.0), 2)
            w = tm.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        d = -tm.log(1.0 - ti.random(ti.f32)) / sigma_t[c]   # free distance

        # --- nearest boundary ahead ---
        t_wall = t_out
        hit_outer = 1
        if t_in < t_out:
            t_wall = t_in
            hit_outer = 0

        if d < t_wall:
            # scattered before reaching a boundary
            trans = tm.exp(-sigma_t * d)
            pdf_vec = sigma_t * trans
            # mixture pdf, weighted the same way the channel was chosen.
            # w sums to 1, so no division by wsum is needed
            pdf = w[0] * pdf_vec[0] + w[1] * pdf_vec[1] + w[2] * pdf_vec[2]
            throughput *= (sigma_s * trans) / tm.max(pdf, 1e-8)
            throughput = tm.min(throughput, MAX_THROUGHPUT)

            pos = pos + d * dir
            dir, phase = sample_hg(dir, G)
            n_scat += 1.0

        else:
            # reached a boundary first; no scattering event, so no sigma_s
            trans = tm.exp(-sigma_t * t_wall)
            pdf = w[0] * trans[0] + w[1] * trans[1] + w[2] * trans[2]
            throughput *= trans / tm.max(pdf, 1e-8)
            throughput = tm.min(throughput, MAX_THROUGHPUT)
            n_wall += 1.0

            pos = pos + t_wall * dir

            if hit_outer == 1:
                # --- outer surface: escape, or reflect back inside? ---
                # front_out is -1 here (struck from within), and passing it to
                # interpolate_normal already flips the normal to face the ray.
                # Negate to get the outward-pointing normal.
                n_exit = -interpolate_normal(tri_out, u_out, v_out, front_out)

                cos_i = tm.clamp(tm.dot(dir, n_exit), 0.0, 1.0)
                # probability the light does NOT get out
                F_exit = fresnel_dielectric_exit(cos_i, IOR_SKIN)

                if ti.random(ti.f32) < F_exit:
                    # trapped. mirror the direction about the surface and
                    # keep walking. choosing this branch with probability
                    # F_exit exactly cancels the F_exit weight, so throughput
                    # is unchanged - and here the two really are the same
                    # number, unlike the entry-side coin in radiance.py
                    dir = dir - 2.0 * tm.dot(dir, n_exit) * n_exit
                    pos = pos - n_exit * SURF_EPS      # nudge back inside
                    ti.atomic_add(walk_tir[None], 1)

                else:
                    escaped = 1
                    exit_tri = tri_out
                    exit_u = u_out
                    exit_v = v_out
                    ti.atomic_add(walk_rgb[0], ti.cast(throughput[0], ti.f64))
                    ti.atomic_add(walk_rgb[1], ti.cast(throughput[1], ti.f64))
                    ti.atomic_add(walk_rgb[2], ti.cast(throughput[2], ti.f64))
                    break

            else:
                # cross the inner boundary. front_in tells us which side we
                # arrived from, so push along the direction of travel to clear
                # it, and carry on the way we were already heading.
                n_cross = interpolate_normal(tri_in, u_in, v_in, front_in)
                # n_cross now opposes dir, so stepping against it clears the
                # surface no matter which side we came from
                pos = pos - n_cross * SURF_EPS
                # no flag to flip: the layer is re-derived next step

        # --- russian roulette: kill dim walks, scale up survivors ---
        # only ever scales up a walk that was at risk of being killed. a walk
        # whose throughput is already at or above 1 is left completely alone.
        if step > ROULETTE_START:
            survive = tm.min(1.0, tm.max(throughput[0],
                             tm.max(throughput[1], throughput[2])))
            if survive < 1.0:
                if ti.random(ti.f32) > survive:
                    rouletted = 1
                    break
                throughput /= survive

    ti.atomic_add(walk_diag[0], ti.cast(n_scat, ti.f64))
    ti.atomic_add(walk_diag[1], ti.cast(n_wall, ti.f64))
    ti.atomic_add(walk_diag[2], ti.cast(n_epi, ti.f64))
    ti.atomic_add(walk_diag[3], ti.cast(n_step, ti.f64))

    if escaped == 1:
        ti.atomic_add(walk_stats[0], 1)
    elif lost == 1:
        ti.atomic_add(walk_stats[2], 1)
    elif rouletted == 1:
        ti.atomic_add(walk_stats[1], 1)     # roulette - unbiased
    else:
        ti.atomic_add(walk_stats[3], 1)     # hard cap - real loss

    return pos, dir, throughput, escaped, exit_tri, exit_u, exit_v