import taichi as ti
import taichi.math as tm
from brdf import ggx_distribution

@ti.func
def build_coordinate_system(n):
    sign = 1.0 if n[2] >= 0.0 else -1.0
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    t = tm.vec3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0])
    bt = tm.vec3(b, sign + n[1] * n[1] * a, -n[1])
    return t, bt


@ti.func
def cosine_weighted_hemisphere_sample(normal):
    tangent, bitangent = build_coordinate_system(normal)

    r1 = ti.random(ti.f32)
    r2 = ti.random(ti.f32)

    sin_theta = tm.sqrt(r1)
    cos_theta = tm.sqrt(tm.max(0.0, 1.0 - r1))
    phi = 2.0 * tm.pi * r2

    x_local = sin_theta * tm.cos(phi)
    y_local = sin_theta * tm.sin(phi)
    z_local = cos_theta

    world_dir = tm.normalize(x_local * tangent + y_local * bitangent + z_local * normal)
    pdf = cos_theta / tm.pi

    return world_dir, pdf



@ti.func
def sample_ggx(n, wo, roughness):
    tangent, bitangent = build_coordinate_system(n)

    alpha = roughness * roughness
    alpha2 = alpha * alpha

    r1 = ti.random(ti.f32)
    r2 = ti.random(ti.f32)

    cos_theta = tm.sqrt((1.0 - r1) / (1.0 + (alpha2 - 1.0) * r1 + 1e-8))
    sin_theta = tm.sqrt(tm.max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * tm.pi * r2

    h = tm.normalize(
        sin_theta * tm.cos(phi) * tangent
        + sin_theta * tm.sin(phi) * bitangent
        + cos_theta * n
    )

    wi = tm.normalize(2.0 * tm.dot(wo, h) * h - wo)

    pdf = 0.0

    if tm.dot(wi, n) > 0.0:
        D = ggx_distribution(n, h, roughness)
        pdf = D * tm.max(0.0, tm.dot(n, h)) / (4.0 * tm.max(1e-6, tm.dot(wo, h)))

    return wi, pdf 
