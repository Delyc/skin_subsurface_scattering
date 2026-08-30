import numpy as np
import taichi as ti
import taichi.math as tm

from texture import _read, srgb_to_linear, TEX_DTYPE, _NP_DTYPE, _texel_coords
from brdf import (ggx_distribution, smith_visibility, fresnel_schlick,
                  smith_g1)

import os as _oo; EYE_TEX = int(_oo.environ.get("EYE_TEX",4096))

eye_diffuse = ti.Vector.field(3, dtype=TEX_DTYPE, shape=(EYE_TEX, EYE_TEX))
eye_normal = ti.Vector.field(3, dtype=TEX_DTYPE, shape=(EYE_TEX, EYE_TEX))
eye_roughness = ti.field(dtype=TEX_DTYPE, shape=(EYE_TEX, EYE_TEX))
eye_specular = ti.field(dtype=TEX_DTYPE, shape=(EYE_TEX, EYE_TEX))


EYE_F0 = 0.025


def _resize_to(img, n):
    h, w = img.shape[:2]
    if (h, w) == (n, n):
        return img
    ys = (np.arange(n) * h / n).astype(np.int32)
    xs = (np.arange(n) * w / n).astype(np.int32)
    return img[np.ix_(ys, xs)] if img.ndim == 2 else img[np.ix_(ys, xs)]


def load_eye_textures(diffuse_path, normal_path, roughness_path, specular_path):
    d = srgb_to_linear(_resize_to(_read(diffuse_path, 3, expect_size=0), EYE_TEX))
    eye_diffuse.from_numpy(d.astype(_NP_DTYPE))

    n = _resize_to(_read(normal_path, 3, expect_size=0), EYE_TEX) * 2.0 - 1.0
    lengths = np.linalg.norm(n, axis=2, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    eye_normal.from_numpy((n / lengths).astype(_NP_DTYPE))

    eye_roughness.from_numpy(
        _resize_to(_read(roughness_path, 1, expect_size=0), EYE_TEX).astype(_NP_DTYPE))
    eye_specular.from_numpy(
        _resize_to(_read(specular_path, 1, expect_size=0), EYE_TEX).astype(_NP_DTYPE))


@ti.func
def _bilinear3_eye(field: ti.template(), uv):
    x = tm.clamp(uv[0], 0.0, 1.0) * EYE_TEX - 0.5
    y = (1.0 - tm.clamp(uv[1], 0.0, 1.0)) * EYE_TEX - 0.5
    x0 = ti.cast(tm.floor(x), ti.i32)
    y0 = ti.cast(tm.floor(y), ti.i32)
    fx = x - x0
    fy = y - y0
    x0 = tm.clamp(x0, 0, EYE_TEX - 1)
    y0 = tm.clamp(y0, 0, EYE_TEX - 1)
    x1 = tm.clamp(x0 + 1, 0, EYE_TEX - 1)
    y1 = tm.clamp(y0 + 1, 0, EYE_TEX - 1)
    top = ti.cast(field[y0, x0], ti.f32) * (1 - fx) + ti.cast(field[y0, x1], ti.f32) * fx
    bot = ti.cast(field[y1, x0], ti.f32) * (1 - fx) + ti.cast(field[y1, x1], ti.f32) * fx
    return top * (1 - fy) + bot * fy


@ti.func
def sample_eye_diffuse(uv):
    return _bilinear3_eye(eye_diffuse, uv)


@ti.func
def sample_eye_roughness(uv):
    return _bilinear3_eye(eye_roughness, uv)


@ti.func
def eval_eye_brdf(n, wi, wo, uv):
    cos_i = tm.max(0.0, tm.dot(n, wi))
    cos_o = tm.max(0.0, tm.dot(n, wo))

    result = tm.vec3(0.0, 0.0, 0.0)
    if cos_i > 1e-4 and cos_o > 1e-4:
        albedo = sample_eye_diffuse(uv)
        rough = tm.max(sample_eye_roughness(uv), 0.05)

        # diffuse lobe, energy split with the specular via 1 - F
        h = tm.normalize(wi + wo)
        F = fresnel_schlick(tm.max(0.0, tm.dot(wo, h)), EYE_F0)
        diffuse = albedo * (1.0 / tm.pi) * (1.0 - F)

        D = ggx_distribution(n, h, rough)
        V = smith_visibility(cos_i, cos_o, rough)
        spec = D * V * F

        result = diffuse + tm.vec3(spec, spec, spec)
    return result