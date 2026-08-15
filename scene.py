import numpy as np
import taichi as ti
import taichi.math as tm
import os
from load_obj import load_obj, triangulate_faces
from bvh import compute_centroids, build_bvh, flatten_bvh
from helpers import (compute_face_normals, compute_vertex_normal, compute_tangents,
                     find_boundary_edges, boundary_center, cap_boundary)

MM_PER_UNIT = 660.0          # mesh is ~0.235 units ear to ear; a head is ~155 mm

# ---------------------------------------------------------------- load mesh
positions, uvs, faces = load_obj("Head.obj")
positions = np.array(positions, dtype=np.float32) * MM_PER_UNIT
faces = triangulate_faces(faces)

# ------------------------------------------------- close the open bottom
# the bust is a shell, open where the chest ends. subsurface rays would
# walk out through that hole and hit nothing, so seal it before anything
# else is derived from the mesh.

def count_loops(boundary):
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    loops = 0
    for v in adj:
        if v in seen:
            continue
        loops += 1
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x])
    return loops


boundary = find_boundary_edges(faces)
print("boundary loops:", count_loops(boundary))   

center, bverts = boundary_center(positions, boundary)
positions, uvs, faces = cap_boundary(positions, uvs, faces, boundary, center)
print("boundary edges after capping:", len(find_boundary_edges(faces)))

# everything below depends on the capped mesh, so derive it only now
uvs_np = np.array(uvs, dtype=np.float32)
num_faces = len(faces)
num_verts = positions.shape[0]
num_uvs = uvs_np.shape[0]

# ---------------------------------------------------------------- tangents
tri_tangent_np = compute_tangents(positions, uvs, faces)
tri_tangent = ti.Vector.field(3, dtype=ti.f32, shape=num_faces)
tri_tangent.from_numpy(tri_tangent_np)

print("degenerate tangents:",
      np.sum(np.all(tri_tangent_np == np.array([1.0, 0.0, 0.0]), axis=1)))

# ---------------------------------------------------------------- normals
centroids = compute_centroids(positions, faces)
face_normals = compute_face_normals(faces, positions)
vertex_normals = compute_vertex_normal(faces, positions, face_normals)

# ------------------------------------------------------------ inner mesh
EPIDERMIS_THICKNESS = float(os.environ.get("SKIN_T", 0.03))
inner_positions = positions - vertex_normals * EPIDERMIS_THICKNESS

inner_face_normals = compute_face_normals(faces, inner_positions)
dots = np.sum(inner_face_normals * face_normals, axis=1)
print("flipped triangles:", np.sum(dots < 0))

# ------------------------------------------------------------ outer BVH
all_indices = np.arange(num_faces)
root = build_bvh(positions, faces, centroids, all_indices)

(node_bbox_min_np, node_bbox_max_np, node_left_np, node_right_np,
 node_tri_start_np, node_tri_count_np, leaf_triangle_indices_np) = flatten_bvh(root)

num_nodes = node_bbox_min_np.shape[0]
num_leaf_refs = leaf_triangle_indices_np.shape[0]

# per-triangle index tables (shared by both meshes - same topology)
tri_vertex_idx_np = np.array(
    [[face[0][0], face[1][0], face[2][0]] for face in faces], dtype=np.int32
)
tri_uv_idx_np = np.array(
    [[face[0][1], face[1][1], face[2][1]] for face in faces], dtype=np.int32
)

# in scene.py
overflow_count = ti.field(dtype=ti.i32, shape=())

# ---------------------------------------------------------- outer fields
positions_field = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
positions_field.from_numpy(positions)

vertex_normals_field = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
vertex_normals_field.from_numpy(vertex_normals.astype(np.float32))

tri_vertex_idx = ti.Vector.field(3, dtype=ti.i32, shape=num_faces)
tri_vertex_idx.from_numpy(tri_vertex_idx_np)

tri_uv_idx = ti.Vector.field(3, dtype=ti.i32, shape=num_faces)
tri_uv_idx.from_numpy(tri_uv_idx_np)

uvs_field = ti.Vector.field(2, dtype=ti.f32, shape=num_uvs)
uvs_field.from_numpy(uvs_np)

node_bbox_min = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
node_bbox_max = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
node_bbox_min.from_numpy(node_bbox_min_np)
node_bbox_max.from_numpy(node_bbox_max_np)

node_left = ti.field(dtype=ti.i32, shape=num_nodes)
node_right = ti.field(dtype=ti.i32, shape=num_nodes)
node_left.from_numpy(node_left_np)
node_right.from_numpy(node_right_np)

node_tri_start = ti.field(dtype=ti.i32, shape=num_nodes)
node_tri_count = ti.field(dtype=ti.i32, shape=num_nodes)
node_tri_start.from_numpy(node_tri_start_np)
node_tri_count.from_numpy(node_tri_count_np)

leaf_triangle_indices = ti.field(dtype=ti.i32, shape=num_leaf_refs)
leaf_triangle_indices.from_numpy(leaf_triangle_indices_np)

# ------------------------------------------------------------ inner BVH
inner_centroids = compute_centroids(inner_positions, faces)
inner_root = build_bvh(inner_positions, faces, inner_centroids, all_indices)

(inner_node_bbox_min_np, inner_node_bbox_max_np, inner_node_left_np,
 inner_node_right_np, inner_node_tri_start_np, inner_node_tri_count_np,
 inner_leaf_triangle_indices_np) = flatten_bvh(inner_root)

inner_num_nodes = inner_node_bbox_min_np.shape[0]
inner_num_leaf_refs = inner_leaf_triangle_indices_np.shape[0]

inner_positions_field = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
inner_positions_field.from_numpy(inner_positions.astype(np.float32))

inner_node_bbox_min = ti.Vector.field(3, dtype=ti.f32, shape=inner_num_nodes)
inner_node_bbox_max = ti.Vector.field(3, dtype=ti.f32, shape=inner_num_nodes)
inner_node_bbox_min.from_numpy(inner_node_bbox_min_np)
inner_node_bbox_max.from_numpy(inner_node_bbox_max_np)

inner_node_left = ti.field(dtype=ti.i32, shape=inner_num_nodes)
inner_node_right = ti.field(dtype=ti.i32, shape=inner_num_nodes)
inner_node_left.from_numpy(inner_node_left_np)
inner_node_right.from_numpy(inner_node_right_np)

inner_node_tri_start = ti.field(dtype=ti.i32, shape=inner_num_nodes)
inner_node_tri_count = ti.field(dtype=ti.i32, shape=inner_num_nodes)
inner_node_tri_start.from_numpy(inner_node_tri_start_np)
inner_node_tri_count.from_numpy(inner_node_tri_count_np)

inner_leaf_triangle_indices = ti.field(dtype=ti.i32, shape=inner_num_leaf_refs)
inner_leaf_triangle_indices.from_numpy(inner_leaf_triangle_indices_np)

print("outer BVH nodes:", num_nodes, " inner BVH nodes:", inner_num_nodes)

# ------------------------------------------------------ traversal scratch
# one stack per pixel PER MESH. the outer and inner traversals run back to
# back inside the subsurface walk, so they must not share scratch space.
STACK_MAX = 64
stack_field = ti.field(dtype=ti.i32, shape=(600, 600, 2, STACK_MAX))


print("mesh min:", positions.min(axis=0))
print("mesh max:", positions.max(axis=0))


i = np.argmin(positions[:, 0])
print("ear tip vertex:", positions[i])

# and the spread of everything near that x, to get the ear's extent
head = positions[positions[:, 1] > 170]
i = np.argmin(head[:, 0])
print("head widest point:", head[i])

near = head[head[:, 0] < head[:, 0].min() + 8]
print("ear y range:", near[:, 1].min(), near[:, 1].max())
print("ear z range:", near[:, 2].min(), near[:, 2].max())

# print("boundary edges:", len(boundary))
# def boundary_center(positions, boundary):
#     verts = set()
#     for a, b in boundary:
#         verts.add(a)
#         verts.add(b)
#     verts = list(verts)
#     return np.mean(positions[verts], axis=0), verts

# center, bverts = boundary_center(positions, boundary)
# print("boundary verts:", len(bverts))
# print("center:", center)

# positions, uvs, faces = cap_boundary(positions, uvs, faces, boundary, center)
# print("faces after capping:", len(faces))

# boundary_after = find_boundary_edges(faces)
# print("boundary edges after capping:", len(boundary_after))