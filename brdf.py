import os

import numpy as np
import taichi as ti
import taichi.math as tm

from texture import sample_specular

SKIN_F0 = 0.028
SPEC_MAP_RANGE = 0.5   # map drives F0 over [0.5, 1.5] * SKIN_F0


@ti.func
def specular_f0(uv):
    return SKIN_F0 * (1.0 - SPEC_MAP_RANGE + 2.0 * SPEC_MAP_RANGE * sample_specular(uv))


@ti.func
def ggx_distribution(n, h, roughness):
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    cos_h = tm.max(0.0, tm.dot(n, h))
    denom = cos_h * cos_h * (alpha2 - 1.0) + 1.0
    return alpha2 / (tm.pi * denom * denom + 1e-8)


@ti.func
def smith_g1(n, v, roughness):
    alpha = roughness * roughness
    k = alpha / 2.0
    cos_v = tm.max(0.0, tm.dot(n, v))
    return cos_v / (cos_v * (1.0 - k) + k + 1e-8)


@ti.func
def smith_geometry(n, wi, wo, roughness):
    return smith_g1(n, wi, roughness) * smith_g1(n, wo, roughness)


@ti.func
def smith_visibility(cos_i, cos_o, roughness):
    alpha2 = (roughness * roughness) ** 2
    lambda_o = cos_i * tm.sqrt(cos_o * cos_o * (1.0 - alpha2) + alpha2)
    lambda_i = cos_o * tm.sqrt(cos_i * cos_i * (1.0 - alpha2) + alpha2)
    return 0.5 / tm.max(lambda_o + lambda_i, 1e-8)


@ti.func
def fresnel_schlick(cos_theta, f0):
    m = tm.clamp(1.0 - cos_theta, 0.0, 1.0)
    return f0 + (1.0 - f0) * (m * m * m * m * m)


# energy compensation

ESS_RES = 32
_ESS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ggx_ess.npy")

ggx_ess = ti.field(dtype=ti.f32, shape=(ESS_RES, ESS_RES))


def _compute_ess(res=ESS_RES, samples=4096):
    rng = np.random.default_rng(0)
    table = np.zeros((res, res), np.float32)
    r1 = (np.arange(samples) + 0.5) / samples

    for ai in range(res):
        alpha = max(((ai + 0.5) / res) ** 2, 1e-4)
        a2 = alpha * alpha
        for ci in range(res):
            cos_o = max((ci + 0.5) / res, 1e-3)
            wo = np.array([np.sqrt(1 - cos_o * cos_o), 0.0, cos_o])

            ct = np.sqrt((1 - r1) / (1 + (a2 - 1) * r1))
            st = np.sqrt(np.maximum(0, 1 - ct * ct))
            phi = 2 * np.pi * rng.random(samples)
            h = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)

            dot_oh = h @ wo
            cos_i = (2 * dot_oh[:, None] * h - wo)[:, 2]

            lo = cos_i * np.sqrt(cos_o * cos_o * (1 - a2) + a2)
            li = cos_o * np.sqrt(cos_i * cos_i * (1 - a2) + a2)
            V = 0.5 / np.maximum(lo + li, 1e-8)

            w = np.where((cos_i > 0) & (dot_oh > 0),
                         4.0 * V * np.maximum(cos_i, 0) * dot_oh
                         / np.maximum(ct, 1e-6), 0.0)
            table[ai, ci] = w.mean()

    return np.clip(table, 1e-3, 1.0)


def load_ess_table():
    if os.path.exists(_ESS_CACHE):
        table = np.load(_ESS_CACHE).astype(np.float32)
    else:
        table = _compute_ess()
        try:
            np.save(_ESS_CACHE, table)
        except OSError:
            pass
    ggx_ess.from_numpy(table)


load_ess_table()


@ti.func
def sample_ess(cos_o, roughness):
    x = tm.clamp(tm.sqrt(roughness * roughness), 0.0, 0.999) * ESS_RES - 0.5
    y = tm.clamp(cos_o, 0.0, 0.999) * ESS_RES - 0.5

    x0 = tm.clamp(ti.cast(tm.floor(x), ti.i32), 0, ESS_RES - 1)
    y0 = tm.clamp(ti.cast(tm.floor(y), ti.i32), 0, ESS_RES - 1)
    x1 = tm.clamp(x0 + 1, 0, ESS_RES - 1)
    y1 = tm.clamp(y0 + 1, 0, ESS_RES - 1)
    fx = tm.clamp(x - x0, 0.0, 1.0)
    fy = tm.clamp(y - y0, 0.0, 1.0)

    top = ggx_ess[x0, y0] * (1.0 - fx) + ggx_ess[x1, y0] * fx
    bot = ggx_ess[x0, y1] * (1.0 - fx) + ggx_ess[x1, y1] * fx
    return tm.max(top * (1.0 - fy) + bot * fy, 1e-3)


@ti.func
def ms_compensation(cos_o, roughness, f0):
    ess = sample_ess(cos_o, roughness)
    return 1.0 + f0 * (1.0 - ess) / ess


@ti.func
def eval_specular_brdf(n, wi, wo, uv, roughness):
    f0 = specular_f0(uv)
    cos_i = tm.max(0.0, tm.dot(n, wi))
    cos_o = tm.max(0.0, tm.dot(n, wo))

    result = tm.vec3(0.0, 0.0, 0.0)
    if cos_i > 1e-4 and cos_o > 1e-4:
        h = tm.normalize(wi + wo)
        D = ggx_distribution(n, h, roughness)
        V = smith_visibility(cos_i, cos_o, roughness)
        Fr = fresnel_schlick(tm.max(0.0, tm.dot(wo, h)), f0)
        s = D * V * Fr * ms_compensation(cos_o, roughness, f0)
        result = tm.vec3(s, s, s)
    return result


@ti.func
def subsurface_weight(n, wo, uv):
    cos_o = tm.max(0.0, tm.dot(n, wo))
    return 1.0 - fresnel_schlick(cos_o, specular_f0(uv))


@ti.func
def brdf_pdf(n, wi, wo, roughness, prob_specular):
    cos_i = tm.max(0.0, tm.dot(n, wi))
    pdf_diffuse = cos_i / tm.pi

    pdf_specular = 0.0
    h = wi + wo
    if tm.dot(h, h) > 1e-8:
        h = tm.normalize(h)
        D = ggx_distribution(n, h, roughness)
        pdf_specular = D * tm.max(0.0, tm.dot(n, h)) / (
            4.0 * tm.max(1e-6, tm.dot(wo, h)))

    return prob_specular * pdf_specular + (1.0 - prob_specular) * pdf_diffuse