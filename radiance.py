import taichi as ti
import taichi.math as tm

from helpers import environment_light
from shading import interpolate_normal, interpolate_uv
from texture import apply_normal_map, sample_roughness
from scene import tri_tangent, tri_handedness
from trace import trace_outer
from sampling import cosine_weighted_hemisphere_sample, sample_ggx
from lighting import (direct_light_specular, direct_light_diffuse,
                      intersect_light, light_pdf_toward, power_heuristic,
                      light_emission)
from brdf import fresnel_schlick, eval_specular_brdf, brdf_pdf
from sss import random_walk_sss, refract_into_medium, F0_SKIN, IOR_SKIN
from scene import HAS_EYES

if HAS_EYES:
    from trace import trace_eyeball
    from scene import (eye_vertex_normals_field, eye_tri_vertex_idx,
                       eye_tri_uv_idx, eye_uvs_field)
    from eye import eval_eye_brdf
    from lighting import sample_light, shadow_ray_blocked

MAX_BOUNCE = 8

# Scene is in millimetres, so offsets are too. 1e-2 mm sits well above f32
# resolution at the far side of a 150 mm head while staying far below any
# feature you would see.
SURF_EPS = 1e-2

# ROUGHNESS = sample_roughness(uv)


if HAS_EYES:
    @ti.func
    def eye_normal_at(tri_idx, u, v, front):
        vi = eye_tri_vertex_idx[tri_idx]
        n = ((1.0 - u - v) * eye_vertex_normals_field[vi[0]]
             + u * eye_vertex_normals_field[vi[1]]
             + v * eye_vertex_normals_field[vi[2]])
        return tm.normalize(n) * front

    @ti.func
    def eye_uv_at(tri_idx, u, v):
        ti_ = eye_tri_uv_idx[tri_idx]
        return ((1.0 - u - v) * eye_uvs_field[ti_[0]]
                + u * eye_uvs_field[ti_[1]]
                + v * eye_uvs_field[ti_[2]])

    @ti.func
    def shade_eyeball(hit_point, tri_idx, u, v, front, wo, px, py):
        """Direct lighting on the eyeball: opaque, diffuse + specular, no walk.
        One NEE sample against the area light with a shadow test."""
        n = eye_normal_at(tri_idx, u, v, front)
        uv = eye_uv_at(tri_idx, u, v)

        Ld = tm.vec3(0.0, 0.0, 0.0)
        wi, dist, emission, pdf = sample_light(hit_point)
        if pdf > 0.0:
            cos_s = tm.dot(wi, n)
            if cos_s > 0.0:
                shadow_o = hit_point + n * SURF_EPS
                if not shadow_ray_blocked(shadow_o, wi, dist, px, py):
                    brdf = eval_eye_brdf(n, wi, wo, uv)
                    Ld = emission * brdf * cos_s / pdf
        return Ld


@ti.func
def radiance(ray_origin, ray_dir, px, py):
    L = tm.vec3(0.0, 0.0, 0.0)
    throughput = tm.vec3(1.0, 1.0, 1.0)

    origin = ray_origin
    direction = ray_dir

    # MIS bookkeeping for the previous bounce: if this ray lands on the
    # emitter, weight it against the light-sampling estimate that already
    # ran at the previous vertex.
    prev_pdf = -1.0          # -1 marks the camera ray, which is unweighted
    prev_prob_spec = 1.0

    for bounce in range(MAX_BOUNCE):
        t, tri_idx, u, v, front = trace_outer(origin, direction, px, py)

        # ---- eyeball: opaque, so if it is nearer than the skin the ray
        # stops there. Shade it directly and end the path. ----
        if ti.static(HAS_EYES):
            et, e_tri, eu, ev, e_front = trace_eyeball(origin, direction, px, py)
            head_t = t if tri_idx >= 0 else 1e30
            if e_tri >= 0 and et < head_t:
                e_hit = origin + et * direction
                # still let a BSDF-sampled ray that grazes the light count
                tl, hitl = intersect_light(origin, direction, et)
                if hitl == 1:
                    w = 1.0
                    if prev_pdf > 0.0:
                        pdf_l = light_pdf_toward(origin, direction, tl)
                        w = power_heuristic(prev_pdf, pdf_l)
                    L += throughput * light_emission[None] * w
                else:
                    L += throughput * shade_eyeball(
                        e_hit, e_tri, eu, ev, e_front, -direction, px, py)
                break

        # ---- did this ray hit the light on its way? (BSDF side of MIS) ----
        surface_t = t if tri_idx >= 0 else 1e30
        t_light, hit_light = intersect_light(origin, direction, surface_t)
        if hit_light == 1:
            w = 1.0
            if prev_pdf > 0.0:
                pdf_l = light_pdf_toward(origin, direction, t_light)
                w = power_heuristic(prev_pdf, pdf_l)
            L += throughput * light_emission[None] * w
            break

        if tri_idx < 0:
            L += throughput * environment_light(direction)
            break

        hit_point = origin + t * direction

        # geometric-side normal: drives ray offsets and refraction
        normal = interpolate_normal(tri_idx, u, v, front)
        uv = interpolate_uv(tri_idx, u, v)

        # shading normal: perturbed by the tangent-space map
        shading_normal = apply_normal_map(
            uv, normal, tri_tangent[tri_idx], tri_handedness[tri_idx] * front)

        roughness = sample_roughness(uv)
        wo = -direction

        # Fresnel picks the lobe: reflect off the oil film, or enter the skin.
        # Clamped so neither branch becomes unsamplable.
        F = tm.clamp(
            fresnel_schlick(tm.max(0.0, tm.dot(wo, shading_normal)), F0_SKIN),
            0.02, 0.98)

        if ti.random(ti.f32) < F:
            # ------------------- surface reflection -------------------
            # Dividing by the selection probability leaves eval_specular_brdf
            # to supply the physical Fresnel; the two are not the same factor.
            throughput /= F

            L += throughput * direct_light_specular(
                hit_point, shading_normal, wo, uv, roughness, 1.0, px, py)

            new_dir, pdf = sample_ggx(shading_normal, wo, roughness)
            if pdf <= 1e-8:
                break

            spec = eval_specular_brdf(
                shading_normal, new_dir, wo, uv, roughness)
            cos_theta = tm.max(0.0, tm.dot(new_dir, shading_normal))
            throughput *= spec * cos_theta / pdf

            prev_pdf = pdf
            origin = hit_point + normal * SURF_EPS
            direction = new_dir

        else:
            # ------------------ subsurface scattering ------------------
            # The physical transmission factor (1 - F) cancels exactly against
            # the selection probability, so the net weight here is 1.
            throughput /= (1.0 - F)

            enter_dir = refract_into_medium(direction, normal, IOR_SKIN)
            enter_pos = hit_point - normal * SURF_EPS       # start inside

            exit_pos, exit_dir, walk_tp, escaped, exit_tri, exit_u, exit_v = \
                random_walk_sss(enter_pos, enter_dir, px, py)

            if escaped == 0:
                break        # absorbed in the medium: the path ends here

            # walk_tp carries the colour - melanin and haemoglobin absorption
            throughput *= walk_tp

            exit_normal = interpolate_normal(exit_tri, exit_u, exit_v, 1.0)
            if tm.dot(exit_normal, exit_dir) < 0.0:
                exit_normal = -exit_normal                  # point outward
            exit_uv = interpolate_uv(exit_tri, exit_u, exit_v)

            L += throughput * direct_light_diffuse(
                exit_pos, exit_normal, exit_uv, px, py)

            new_dir, pdf = cosine_weighted_hemisphere_sample(exit_normal)
            if pdf <= 1e-8:
                break

            # Lambertian (1/pi) * cos / (cos/pi) == 1, so throughput is
            # unchanged. No exit Fresnel here: random_walk_sss already spent
            # it on the escape roulette.

            prev_pdf = pdf
            origin = exit_pos + exit_normal * SURF_EPS
            direction = new_dir

        # ---- russian roulette: kill dim paths, scale up the survivors ----
        if bounce > 3:
            survive = tm.min(0.95, tm.max(throughput[0],
                                          tm.max(throughput[1],
                                                 throughput[2])))
            if ti.random(ti.f32) > survive:
                break
            throughput /= survive
            prev_pdf *= survive if prev_pdf > 0.0 else 1.0

    return L



