import numpy as np
import taichi as ti
import taichi.math as tm

from trace import trace_outer
from brdf import eval_specular_brdf

SURF_EPS = 1e-3

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
    area = np.linalg.norm(n)

    light_position[None] = position
    light_u[None] = u_vec
    light_v[None] = v_vec
    light_normal[None] = n / area
    light_emission[None] = emission
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
    if cos_l > 1e-6:                      # zero if p is behind the light
        pdf = dist2 / (cos_l * light_area[None])   # area pdf -> solid angle

    return wi, dist, light_emission[None], pdf


@ti.func
def shadow_ray_blocked(origin, direction, max_dist, px, py):
    t, tri_idx, u, v = trace_outer(origin, direction, px, py)
    return tri_idx >= 0 and t < max_dist - SURF_EPS


@ti.func
def direct_light_specular(hit_point, normal, wo, uv, roughness, px, py):
    Ld = tm.vec3(0.0, 0.0, 0.0)

    wi, dist, emission, pdf = sample_light(hit_point)

    if pdf > 0.0:
        cos_s = tm.dot(wi, normal)
        if cos_s > 0.0:
            shadow_origin = hit_point + normal * SURF_EPS
            if not shadow_ray_blocked(shadow_origin, wi, dist, px, py):
                brdf = eval_specular_brdf(normal, wi, wo, uv, roughness)
                Ld = emission * brdf * cos_s / pdf

    return Ld


@ti.func
def direct_light_diffuse(exit_point, normal, px, py):

    Ld = tm.vec3(0.0, 0.0, 0.0)

    wi, dist, emission, pdf = sample_light(exit_point)

    if pdf > 0.0:
        cos_s = tm.dot(wi, normal)
        if cos_s > 0.0:
            shadow_origin = exit_point + normal * SURF_EPS
            if not shadow_ray_blocked(shadow_origin, wi, dist, px, py):
                Ld = emission * (1.0 / tm.pi) * cos_s / pdf

    return Ld