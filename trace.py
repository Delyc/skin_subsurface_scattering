import numpy as np
import taichi as ti
import taichi.math as tm
from intersect import ray_box_intersect, ray_triangle_intersection

from scene import (positions_field, tri_vertex_idx, node_bbox_min, node_bbox_max,
                   node_left, node_right, node_tri_start, node_tri_count,
                   leaf_triangle_indices, stack_field, STACK_MAX,
                   inner_positions_field, inner_node_bbox_min, inner_node_bbox_max,
                   inner_node_left, inner_node_right, inner_node_tri_start,
                   inner_node_tri_count, inner_leaf_triangle_indices, overflow_count)


SLOT_OUTER = 0
SLOT_INNER = 1


@ti.func
def trace(ray_origin, ray_dir, px, py, slot,
          positions_f: ti.template(),
          bbox_min_f: ti.template(), bbox_max_f: ti.template(),
          left_f: ti.template(), right_f: ti.template(),
          tri_start_f: ti.template(), tri_count_f: ti.template(),
          leaf_tri_f: ti.template()):

    inv_ray_dir = 1.0 / ray_dir
    closest_t = 1e30
    closest_tri = -1
    closest_u = 0.0
    closest_v = 0.0

    stack_field[px, py, slot, 0] = 0
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_id = stack_field[px, py, slot, stack_ptr]

        box_hit = ray_box_intersect(ray_origin, inv_ray_dir,
                                    bbox_min_f[node_id], bbox_max_f[node_id])

        if box_hit < closest_t:
            if tri_count_f[node_id] > 0:
                start = tri_start_f[node_id]
                count = tri_count_f[node_id]
                i = 0
                while i < count:
                    tri_idx = leaf_tri_f[start + i]
                    v_idx = tri_vertex_idx[tri_idx]
                    p0 = positions_f[v_idx[0]]
                    p1 = positions_f[v_idx[1]]
                    p2 = positions_f[v_idx[2]]

                    t, u, v = ray_triangle_intersection(ray_origin, ray_dir,
                                                        p0, p1, p2)
                    if t < closest_t:
                        closest_t = t
                        closest_tri = tri_idx
                        closest_u = u
                        closest_v = v
                    i += 1
                # in trace.py, import it, then:
            else:
                if stack_ptr < STACK_MAX - 1:
                    stack_field[px, py, slot, stack_ptr] = left_f[node_id]
                    stack_ptr += 1
                else:
                    ti.atomic_add(overflow_count[None], 1)
                if stack_ptr < STACK_MAX - 1:
                    stack_field[px, py, slot, stack_ptr] = right_f[node_id]
                    stack_ptr += 1
                else:
                    ti.atomic_add(overflow_count[None], 1)

    return closest_t, closest_tri, closest_u, closest_v


@ti.func
def trace_outer(ray_origin, ray_dir, px, py):
    return trace(ray_origin, ray_dir, px, py, SLOT_OUTER,
                 positions_field, node_bbox_min, node_bbox_max,
                 node_left, node_right, node_tri_start, node_tri_count,
                 leaf_triangle_indices)


@ti.func
def trace_inner(ray_origin, ray_dir, px, py):
    return trace(ray_origin, ray_dir, px, py, SLOT_INNER,
                 inner_positions_field, inner_node_bbox_min, inner_node_bbox_max,
                 inner_node_left, inner_node_right, inner_node_tri_start,
                 inner_node_tri_count, inner_leaf_triangle_indices)