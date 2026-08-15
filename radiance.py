import taichi as ti
import taichi.math as tm

from helpers import environment_light
from shading import interpolate_normal, interpolate_uv, apply_normal_map
from scene import tri_tangent
from trace import trace_outer
from sampling import cosine_weighted_hemisphere_sample, sample_ggx
from lighting import direct_light_specular, direct_light_diffuse
from texture import sample_roughness
from brdf import fresnel_schlick, eval_specular_brdf
from sss import random_walk_sss, refract_into_medium, F0_SKIN, IOR_SKIN

MAX_BOUNCE = 8

# scene is in millimetres, so offsets are in mm too.
# 1e-3 mm is a thousandth of the epidermis thickness - tiny, but far enough
# out that a trace from here will not re-detect the surface we just left.
SURF_EPS = 1e-3


@ti.func
def radiance(ray_origin, ray_dir, px, py):
    L = tm.vec3(0.0, 0.0, 0.0)
    throughput = tm.vec3(1.0, 1.0, 1.0)

    origin = ray_origin
    direction = ray_dir

    for bounce in range(MAX_BOUNCE):
        t, tri_idx, u, v = trace_outer(origin, direction, px, py)

        if tri_idx < 0:
            L += throughput * environment_light(direction)
            break

        hit_point = origin + t * direction

        # geometric normal: used only for ray offsets and refraction
        normal = interpolate_normal(tri_idx, u, v)
        if tm.dot(normal, direction) > 0.0:
            normal = -normal

        uv = interpolate_uv(tri_idx, u, v)

        # shading normal: perturbed by the normal map, used for shading
        shading_normal = apply_normal_map(normal, tri_tangent[tri_idx], uv)

        # roughness = sample_roughness(uv)
        # roughness = 0.25
        roughness = 0.15
        wo = -direction

        # Fresnel decides: bounce off the surface, or go inside?
        # ~3% reflects at normal incidence, rising toward grazing angles.
        F = tm.clamp(fresnel_schlick(tm.max(0.0, tm.dot(wo, shading_normal)), F0_SKIN),
             0.02, 0.98)

        if ti.random(ti.f32) < F:
            # ---------------- surface reflection ----------------
        

            throughput /= F  

            #nee
            L += throughput * direct_light_specular(
                hit_point, shading_normal, wo, uv, roughness, px, py)

            new_dir, pdf = sample_ggx(shading_normal, wo, roughness)
            if pdf <= 1e-8:
                break

            spec = eval_specular_brdf(shading_normal, new_dir, wo, uv, roughness)
            cos_theta = tm.max(0.0, tm.dot(new_dir, shading_normal))
            throughput *= spec * cos_theta / pdf

            origin = hit_point + normal * SURF_EPS
            direction = new_dir

        else:
            # ---------------- subsurface scattering ----------------
            # refract about the GEOMETRIC normal so the entry direction stays
            # consistent with the offset below
            throughput /= (1.0 - F)  
            enter_dir = refract_into_medium(direction, normal, IOR_SKIN)
            enter_pos = hit_point - normal * SURF_EPS     # minus: start inside

            exit_pos, exit_dir, walk_tp, escaped, exit_tri, exit_u, exit_v = \
                random_walk_sss(enter_pos, enter_dir, px, py)

            if escaped == 0:
                break            # walk failed; this drops the path's energy

            # walk_tp carries the colour: melanin and hemoglobin absorption
            throughput *= walk_tp

            exit_normal = interpolate_normal(exit_tri, exit_u, exit_v)
            if tm.dot(exit_normal, exit_dir) < 0.0:
                exit_normal = -exit_normal               # point outward

            # light leaves diffusely; no albedo here, the walk already
            # supplied the colour
            L += throughput * direct_light_diffuse(exit_pos, exit_normal, px, py)

            new_dir, pdf = cosine_weighted_hemisphere_sample(exit_normal)
            if pdf <= 1e-8:
                break
            # Lambertian (1/pi) * cos / (cos/pi) == 1, so throughput is unchanged

            origin = exit_pos + exit_normal * SURF_EPS
            direction = new_dir

        # --- russian roulette: kill dim paths, scale up survivors ---
        if bounce > 3:
            survive = tm.min(0.95, tm.max(throughput[0],
                                          tm.max(throughput[1], throughput[2])))
            if ti.random(ti.f32) > survive:
                break
            throughput /= survive

    return L