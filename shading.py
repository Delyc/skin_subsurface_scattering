import taichi as ti
import taichi.math as tm
from scene import tri_vertex_idx, vertex_normals_field, tri_uv_idx, uvs_field
from texture import sample_normal

@ti.func
def interpolate_normal(tri_idx, u, v):
    v_idx = tri_vertex_idx[tri_idx]
    n0 = vertex_normals_field[v_idx[0]]
    n1 = vertex_normals_field[v_idx[1]]
    n2 = vertex_normals_field[v_idx[2]]

    return tm.normalize((1.0 - u - v) * n0 + u * n1 + v * n2)


@ti.func
def interpolate_uv(tri_idx, u, v):
    uv_idx = tri_uv_idx[tri_idx]
    #uvs for 3 corners
    uv0 = uvs_field[uv_idx[0]]
    uv1 = uvs_field[uv_idx[1]]
    uv2 = uvs_field[uv_idx[2]]

    return (1.0 - u - v) * uv0 + u * uv1 + v * uv2


@ti.func
def apply_normal_map(n, tangent, uv):
    t = tm.normalize(tangent - n * tm.dot(n, tangent))
    b = tm.cross(n, t)

    # decode [0,1] -> [-1,1]
    sampled = sample_normal(uv) * 2.0 - 1.0

    return tm.normalize(sampled[0] * t + sampled[1] * b + sampled[2] * n)