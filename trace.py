import taichi as ti
import taichi.math as tm

from intersect import (ray_box_intersect, ray_triangle_intersect,
                       safe_inv_dir, T_MAX)
from scene import (outer, inner, tri_vertex_idx, stack_field, STACK_MAX,
                   overflow_count)

SLOT_OUTER = 0
SLOT_INNER = 1


@ti.func
def trace(ray_origin, ray_dir, px, py, slot, t_max,
          positions_f: ti.template(),
          bbox_min_f: ti.template(), bbox_max_f: ti.template(),
          left_f: ti.template(), right_f: ti.template(),
          tri_start_f: ti.template(), tri_count_f: ti.template(),
          leaf_tri_f: ti.template()):
    """Closest hit against one shell.

    Returns (t, tri_idx, u, v, front). `front` is 1 when the ray struck the
    outside of the surface and -1 from the inside, which is what the caller
    needs to know whether it is entering or leaving a medium. tri_idx is -1
    on a miss."""
    inv_ray_dir = safe_inv_dir(ray_dir)

    closest_t = t_max
    closest_tri = -1
    closest_u = 0.0
    closest_v = 0.0

    stack_field[px, py, slot, 0] = 0
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_id = stack_field[px, py, slot, stack_ptr]

        if tri_count_f[node_id] > 0:
            start = tri_start_f[node_id]
            for i in range(tri_count_f[node_id]):
                tri_idx = leaf_tri_f[start + i]
                v_idx = tri_vertex_idx[tri_idx]

                hit = ray_triangle_intersect(
                    ray_origin, ray_dir,
                    positions_f[v_idx[0]],
                    positions_f[v_idx[1]],
                    positions_f[v_idx[2]],
                    closest_t)

                if hit[0] < closest_t:
                    closest_t = hit[0]
                    closest_tri = tri_idx
                    closest_u = hit[1]
                    closest_v = hit[2]
        else:
            left = left_f[node_id]
            right = right_f[node_id]

            t_left = ray_box_intersect(ray_origin, inv_ray_dir,
                                       bbox_min_f[left], bbox_max_f[left],
                                       closest_t)
            t_right = ray_box_intersect(ray_origin, inv_ray_dir,
                                        bbox_min_f[right], bbox_max_f[right],
                                        closest_t)

            # Push far child first so the near one pops first. Visiting front
            # to back lets closest_t tighten early and cull the rest.
            near, far = left, right
            t_near, t_far = t_left, t_right
            if t_right < t_left:
                near, far = right, left
                t_near, t_far = t_right, t_left

            if t_far < closest_t:
                if stack_ptr < STACK_MAX:
                    stack_field[px, py, slot, stack_ptr] = far
                    stack_ptr += 1
                else:
                    ti.atomic_add(overflow_count[None], 1)
            if t_near < closest_t:
                if stack_ptr < STACK_MAX:
                    stack_field[px, py, slot, stack_ptr] = near
                    stack_ptr += 1
                else:
                    ti.atomic_add(overflow_count[None], 1)

    # Which side did we land on? Geometric normal, not the shading normal:
    # medium tracking has to follow the actual surface being crossed.
    front = 0.0
    if closest_tri >= 0:
        v_idx = tri_vertex_idx[closest_tri]
        p0 = positions_f[v_idx[0]]
        n = tm.cross(positions_f[v_idx[1]] - p0, positions_f[v_idx[2]] - p0)
        front = -1.0 if tm.dot(n, ray_dir) > 0.0 else 1.0

    return closest_t, closest_tri, closest_u, closest_v, front


@ti.func
def trace_outer(ray_origin, ray_dir, px, py, t_max=T_MAX):
    return trace(ray_origin, ray_dir, px, py, SLOT_OUTER, t_max,
                 outer["positions"], outer["bbox_min"], outer["bbox_max"],
                 outer["left"], outer["right"], outer["tri_start"],
                 outer["tri_count"], outer["leaf_idx"])


@ti.func
def trace_inner(ray_origin, ray_dir, px, py, t_max=T_MAX):
    return trace(ray_origin, ray_dir, px, py, SLOT_INNER, t_max,
                 inner["positions"], inner["bbox_min"], inner["bbox_max"],
                 inner["left"], inner["right"], inner["tri_start"],
                 inner["tri_count"], inner["leaf_idx"])