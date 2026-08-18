import numpy as np
import taichi as ti
import taichi.math as tm

from trace import trace_outer
from brdf import eval_specular_brdf, brdf_pdf


SURF_EPS = 1e-2

light_position = ti.Vector.field(3, dtype=ti.f32, shape=())
light_u = ti.Vector.field(3, dtype=ti.f32, shape=())
light_v = ti.Vector.field(3, dtype=ti.f32, shape=())
light_normal = ti.Vector.field(3, dtype=ti.f32, shape=())
light_emission = ti.Vector.field(3, dtype=ti.f32, shape=())
light_area = ti.field(dtype=ti.f32, shape=())


def set_light(position, u_vec, v_vec, emission):
    u = np.array(u_vec, dtype=np.float32)
    v = np.array(v_vec, dtype=np.float32)
    n = np.cross(u, v)
    area = float(np.linalg.norm(n))
    if area < 1e-12:
        raise ValueError("degenerate light: u and v are parallel")

    light_position[None] = list(position)
    light_u[None] = u.tolist()
    light_v[None] = v.tolist()
    light_normal[None] = (n / area).tolist()
    light_emission[None] = list(emission)
    light_area[None] = area


@ti.func
def sample_light(p):
    u = ti.random(ti.f32)
    v = ti.random(ti.f32)
    q = light_position[None] + u * light_u[None] + v * light_v[None]

    to_light = q - p
    dist2 = tm.dot(to_light, to_light)
    dist = tm.sqrt(dist2)
    wi = to_light / dist

    cos_l = tm.dot(-wi, light_normal[None])

    pdf = 0.0
    if cos_l > 1e-6:                              # one sided emitter
        pdf = dist2 / (cos_l * light_area[None])  # area pdf -> solid angle

    return wi, dist, light_emission[None], pdf


@ti.func
def light_pdf_toward(p, wi, dist):
    cos_l = tm.dot(-wi, light_normal[None])
    pdf = 0.0
    if cos_l > 1e-6:
        pdf = (dist * dist) / (cos_l * light_area[None])
    return pdf


@ti.func
def shadow_ray_blocked(origin, direction, max_dist, px, py):
    t, tri_idx, u, v, front = trace_outer(
        origin, direction, px, py, max_dist - SURF_EPS)
    return tri_idx >= 0


@ti.func
def power_heuristic(pdf_a, pdf_b):
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    return a2 / tm.max(a2 + b2, 1e-12)


@ti.func
def direct_light_specular(hit_point, normal, wo, uv, roughness,
                          prob_specular, px, py):
    Ld = tm.vec3(0.0, 0.0, 0.0)

    wi, dist, emission, pdf_light = sample_light(hit_point)

    if pdf_light > 0.0:
        cos_s = tm.dot(wi, normal)
        if cos_s > 0.0:
            shadow_origin = hit_point + normal * SURF_EPS
            if not shadow_ray_blocked(shadow_origin, wi, dist, px, py):
                brdf = eval_specular_brdf(normal, wi, wo, uv, roughness)
                pdf_brdf = brdf_pdf(normal, wi, wo, roughness, prob_specular)
                w = power_heuristic(pdf_light, pdf_brdf)
                Ld = emission * brdf * cos_s * w / pdf_light

    return Ld


@ti.func
def direct_light_diffuse(exit_point, normal, uv, px, py):

    Ld = tm.vec3(0.0, 0.0, 0.0)

    wi, dist, emission, pdf = sample_light(exit_point)

    if pdf > 0.0:
        cos_s = tm.dot(wi, normal)
        if cos_s > 0.0:
            shadow_origin = exit_point + normal * SURF_EPS
            if not shadow_ray_blocked(shadow_origin, wi, dist, px, py):
                pdf_brdf = cos_s / tm.pi
                w = power_heuristic(pdf, pdf_brdf)
                Ld = emission * (1.0 / tm.pi) * cos_s * w / pdf

    return Ld


@ti.func
def intersect_light(origin, direction, t_max):

    t = -1.0
    hit = 0

    denom = tm.dot(direction, light_normal[None])
    if abs(denom) > 1e-9:
        rel = light_position[None] - origin
        t_plane = tm.dot(rel, light_normal[None]) / denom
        if 1e-4 < t_plane < t_max:
            q = origin + t_plane * direction - light_position[None]
            u_vec = light_u[None]
            v_vec = light_v[None]
            a = tm.dot(q, u_vec) / tm.dot(u_vec, u_vec)
            b = tm.dot(q, v_vec) / tm.dot(v_vec, v_vec)
            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                t = t_plane
                hit = 1

    return t, hit