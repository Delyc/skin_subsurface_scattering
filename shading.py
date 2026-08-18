import taichi as ti
import taichi.math as tm

from scene import (tri_vertex_idx, vertex_normals_field, tri_uv_idx,
                   uvs_field, tri_tangent, tri_handedness)
from texture import apply_normal_map


@ti.func
def interpolate_normal(tri_idx, u, v, front):
    """Smooth shading normal at a barycentric point.

    `front` comes from trace: +1 when the ray hit the outside, -1 from the
    inside. Subsurface rays strike the surface from within constantly, and
    there the interpolated normal faces away from the ray, so every dot
    product downstream would carry the wrong sign unless it is flipped."""
    v_idx = tri_vertex_idx[tri_idx]
    n0 = vertex_normals_field[v_idx[0]]
    n1 = vertex_normals_field[v_idx[1]]
    n2 = vertex_normals_field[v_idx[2]]

    n = (1.0 - u - v) * n0 + u * n1 + v * n2
    return tm.normalize(n) * front


@ti.func
def interpolate_uv(tri_idx, u, v):
    uv_idx = tri_uv_idx[tri_idx]
    uv0 = uvs_field[uv_idx[0]]
    uv1 = uvs_field[uv_idx[1]]
    uv2 = uvs_field[uv_idx[2]]
    return (1.0 - u - v) * uv0 + u * uv1 + v * uv2


@ti.func
def shading_frame(tri_idx, u, v, front, use_normal_map):
    """Everything the BSDF needs at a hit: perturbed normal and UV.

    The tangent is per face and the handedness sign goes with it; a head
    unwrap is mirrored across the centre line, so without the sign the
    bitangent is backwards on one side of the face."""
    uv = interpolate_uv(tri_idx, u, v)
    N = interpolate_normal(tri_idx, u, v, front)

    if use_normal_map == 1:
        N = apply_normal_map(uv, N, tri_tangent[tri_idx],
                             tri_handedness[tri_idx] * front)

    return N, uv