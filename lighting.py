import numpy as np
import taichi as ti
import taichi.math as tm

from trace import trace_outer
from brdf import eval_specular_brdf, brdf_pdf


SURF_EPS = 1e-2
MAX_LIGHTS = 8

light_position = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_u = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_v = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_normal = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_emission = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_area = ti.field(dtype=ti.f32, shape=MAX_LIGHTS)
num_lights = ti.field(dtype=ti.i32, shape=())


def set_lights(lights):
    """lights: list of dicts, each {position, u_vec, v_vec, emission}."""
    n = len(lights)
    if n == 0 or n > MAX_LIGHTS:
        raise ValueError(f"need 1..{MAX_LIGHTS} lights, got {n}")

    for i, L in enumerate(lights):
        u = np.array(L["u_vec"], dtype=np.float32)
        v = np.array(L["v_vec"], dtype=np.float32)
        nrm = np.cross(u, v)
        area = float(np.linalg.norm(nrm))
        if area < 1e-12:
            raise ValueError(f"light {i}: u and v are parallel")

        light_position[i] = list(L["position"])
        light_u[i] = u.tolist()
        light_v[i] = v.tolist()
        light_normal[i] = (nrm / area).tolist()
        light_emission[i] = list(L["emission"])
        light_area[i] = area

    num_lights[None] = n


def set_light(position, u_vec, v_vec, emission):
    """Back-compatible single-light entry point."""
    set_lights([{"position": position, "u_vec": u_vec,
                 "v_vec": v_vec, "emission": emission}])


@ti.func
def sample_light(p):
    """Pick one light uniformly"""
    n = num_lights[None]
    idx = tm.min(ti.cast(ti.random(ti.f32) * n, ti.i32), n - 1)

    u = ti.random(ti.f32)
    v = ti.random(ti.f32)
    q = light_position[idx] + u * light_u[idx] + v * light_v[idx]

    to_light = q - p
    dist2 = tm.dot(to_light, to_light)
    dist = tm.sqrt(dist2)
    wi = to_light / dist

    cos_l = tm.dot(-wi, light_normal[idx])

    pdf = 0.0
    if cos_l > 1e-6:
        # area pdf -> solid angle, times the 1/N chance of picking this light
        pdf = dist2 / (cos_l * light_area[idx] * n)

    return wi, dist, light_emission[idx], pdf


@ti.func
def light_pdf_toward(p, wi, dist):
    """Combined solid-angle pdf of sampling direction wi across ALL lights."""
    n = num_lights[None]
    total = 0.0
    for i in range(n):
        denom = tm.dot(wi, light_normal[i])
        if abs(denom) > 1e-9:
            rel = light_position[i] - p
            t_plane = tm.dot(rel, light_normal[i]) / denom
            if t_plane > 1e-4:
                q = p + t_plane * wi - light_position[i]
                a = tm.dot(q, light_u[i]) / tm.dot(light_u[i], light_u[i])
                b = tm.dot(q, light_v[i]) / tm.dot(light_v[i], light_v[i])
                if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                    cos_l = tm.dot(-wi, light_normal[i])
                    if cos_l > 1e-6:
                        total += (t_plane * t_plane) / (
                            cos_l * light_area[i] * n)
    return total


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
    """Nearest emitter hit across all lights. Returns (t, hit, which) so the
    caller can read that light's emission via light_emission_at(which)."""
    t = -1.0
    hit = 0
    which = -1

    for i in range(num_lights[None]):
        denom = tm.dot(direction, light_normal[i])
        if abs(denom) > 1e-9:
            rel = light_position[i] - origin
            t_plane = tm.dot(rel, light_normal[i]) / denom
            upper = t_max if hit == 0 else t
            if 1e-4 < t_plane < upper:
                q = origin + t_plane * direction - light_position[i]
                a = tm.dot(q, light_u[i]) / tm.dot(light_u[i], light_u[i])
                b = tm.dot(q, light_v[i]) / tm.dot(light_v[i], light_v[i])
                if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                    t = t_plane
                    hit = 1
                    which = i

    return t, hit, which


@ti.func
def light_emission_at(idx):
    """Emission of light `idx`, or zero for the -1 (no-hit) sentinel."""
    e = tm.vec3(0.0, 0.0, 0.0)
    if idx >= 0:
        e = light_emission[idx]
    return e