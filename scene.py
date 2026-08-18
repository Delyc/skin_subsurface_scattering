import os

import numpy as np
import taichi as ti

# This module creates fields at import time, so the runtime must already be
# up. Guarding here means import order can't silently break the build.
if ti.lang.impl.get_runtime().prog is None:
    ti.init(arch=ti.gpu if ti._lib.core.with_cuda() else ti.cpu)

from load_obj import load_obj, triangulate_faces
from bvh import compute_triangle_bboxes, compute_centroids, build_bvh, flatten_bvh
from helpers import (compute_face_normals, compute_vertex_normals,
                     compute_tangents, close_mesh, signed_volume)

# Mesh is ~0.235 units ear to ear; a head is ~155 mm.
MM_PER_UNIT = 660.0

# Epidermis shell depth, in mm. Real facial epidermis is 0.05-0.15 mm.
EPIDERMIS_THICKNESS = float(os.environ.get("SKIN_T", 0.1))

import os as _os
RESOLUTION = (int(_os.environ.get("RENDER_W", 600)),
              int(_os.environ.get("RENDER_H", 600)))
STACK_MAX = 32          # BVH depth is ~14; 32 is already generous

# ------------------------------------------------------------------ mesh
positions, uvs, faces = load_obj("Head.obj")
positions = positions * MM_PER_UNIT
faces = triangulate_faces(faces)

# Subsurface rays would walk out through any hole and hit nothing, so seal
# the mesh before deriving anything from it. Each loop gets its own centre.
positions, uvs, faces = close_mesh(positions, uvs, faces)

volume = signed_volume(positions, faces)
print(f"signed volume: {volume:+.1f} mm^3")
if volume < 0:
    raise RuntimeError(
        "winding is inside out - every normal points into the skull. "
        "Reverse each triangle at load time.")

num_faces = len(faces)
num_verts = positions.shape[0]
num_uvs = uvs.shape[0]

# --------------------------------------------------------------- normals
face_normals = compute_face_normals(positions, faces)
vertex_normals = compute_vertex_normals(positions, faces)
centroids = compute_centroids(positions, faces)

tri_tangent_np, tri_handedness_np = compute_tangents(
    positions, uvs, faces, vertex_normals)

degenerate = np.sum(np.all(tri_tangent_np == np.array([1.0, 0.0, 0.0]), axis=1))
print(f"degenerate tangents: {degenerate} / {num_faces}")

# ---------------------------------------------------------- inner shell
inner_positions = (positions - vertex_normals * EPIDERMIS_THICKNESS).astype(np.float32)

inner_face_normals = compute_face_normals(inner_positions, faces)
flipped = int(np.sum(np.sum(inner_face_normals * face_normals, axis=1) < 0))
print(f"inverted inner triangles: {flipped} / {num_faces}")
if flipped:
    print("  -> the offset exceeds the local concave radius somewhere "
          "(nostrils, ear canal, tear ducts). Reduce SKIN_T.")

inner_volume = signed_volume(inner_positions, faces)
print(f"inner volume: {inner_volume:+.1f} mm^3 "
      f"({100.0 * inner_volume / volume:.1f}% of outer)")

# ------------------------------------------------- shared index tables
# Both shells have identical topology, so these are shared.
tri_vertex_idx_np = np.array(
    [[f[0][0], f[1][0], f[2][0]] for f in faces], dtype=np.int32)

assert all(i is not None for f in faces for (_, i) in f), "missing UV index"
tri_uv_idx_np = np.array(
    [[f[0][1], f[1][1], f[2][1]] for f in faces], dtype=np.int32)


def build_and_upload(verts, label, mesh_faces=None):
    """Build a BVH over `verts` and push it to Taichi fields. Defaults to the
    head's face list; pass mesh_faces for a different mesh (the eyeballs)."""
    if mesh_faces is None:
        mesh_faces = faces
    n_verts = verts.shape[0]
    n_tris = len(mesh_faces)

    tri_min, tri_max = compute_triangle_bboxes(verts, mesh_faces)
    cents = compute_centroids(verts, mesh_faces)
    root = build_bvh(tri_min, tri_max, cents, np.arange(n_tris))

    (bmin, bmax, left, right, tri_start, tri_count, leaf_idx) = flatten_bvh(root)
    n_nodes = bmin.shape[0]

    f = {
        "positions": ti.Vector.field(3, ti.f32, shape=n_verts),
        "bbox_min": ti.Vector.field(3, ti.f32, shape=n_nodes),
        "bbox_max": ti.Vector.field(3, ti.f32, shape=n_nodes),
        "left": ti.field(ti.i32, shape=n_nodes),
        "right": ti.field(ti.i32, shape=n_nodes),
        "tri_start": ti.field(ti.i32, shape=n_nodes),
        "tri_count": ti.field(ti.i32, shape=n_nodes),
        "leaf_idx": ti.field(ti.i32, shape=leaf_idx.shape[0]),
    }
    f["positions"].from_numpy(verts.astype(np.float32))
    f["bbox_min"].from_numpy(bmin)
    f["bbox_max"].from_numpy(bmax)
    f["left"].from_numpy(left)
    f["right"].from_numpy(right)
    f["tri_start"].from_numpy(tri_start)
    f["tri_count"].from_numpy(tri_count)
    f["leaf_idx"].from_numpy(leaf_idx)

    print(f"{label} BVH: {n_nodes} nodes, {leaf_idx.shape[0]} leaf refs")
    return f


outer = build_and_upload(positions, "outer")
inner = build_and_upload(inner_positions, "inner")

# ----------------------------------------------------- shared geometry
vertex_normals_field = ti.Vector.field(3, ti.f32, shape=num_verts)
vertex_normals_field.from_numpy(vertex_normals.astype(np.float32))

tri_vertex_idx = ti.Vector.field(3, ti.i32, shape=num_faces)
tri_vertex_idx.from_numpy(tri_vertex_idx_np)

tri_uv_idx = ti.Vector.field(3, ti.i32, shape=num_faces)
tri_uv_idx.from_numpy(tri_uv_idx_np)

uvs_field = ti.Vector.field(2, ti.f32, shape=num_uvs)
uvs_field.from_numpy(uvs.astype(np.float32))

tri_tangent = ti.Vector.field(3, ti.f32, shape=num_faces)
tri_tangent.from_numpy(tri_tangent_np)

tri_handedness = ti.field(ti.f32, shape=num_faces)
tri_handedness.from_numpy(tri_handedness_np)

# ------------------------------------------------- traversal scratch
# One stack per pixel per mesh. Slot 0 outer, 1 inner, 2 eyeball. The head's
# outer/inner run back to back inside the walk, and the eyeball trace runs
# alongside the primary head trace, so all three need separate scratch.
NUM_MESHES = 3
stack_field = ti.field(ti.i32,
                       shape=(RESOLUTION[0], RESOLUTION[1], NUM_MESHES, STACK_MAX))
overflow_count = ti.field(ti.i32, shape=())

lo, hi = positions.min(axis=0), positions.max(axis=0)
print(f"bounds (mm): {lo} .. {hi}")
print(f"extent (mm): {hi - lo}")


# ============================================================ eyeballs
# Both eyeballs share one texture set and one material, so they merge into a
# single mesh with one BVH. They are opaque - no inner shell, no walk.
def _load_eyeball(path):
    p, uv, f = load_obj(path)
    p = p * MM_PER_UNIT
    f = triangulate_faces(f)
    return p, uv, f


def _merge(meshes):
    """Concatenate several (positions, uvs, faces) into one mesh, offsetting
    the vertex and UV indices so each sub-mesh keeps pointing at its own data."""
    all_p, all_uv, all_f = [], [], []
    v_off, uv_off = 0, 0
    for p, uv, f in meshes:
        all_p.append(p)
        all_uv.append(uv)
        for tri in f:
            all_f.append(tuple((v + v_off, (t + uv_off) if t is not None else None)
                               for (v, t) in tri))
        v_off += len(p)
        uv_off += len(uv)
    return (np.concatenate(all_p), np.concatenate(all_uv), all_f)


try:
    eye_positions, eye_uvs, eye_faces = _merge([
        _load_eyeball("OBJ/eyeball_L.obj"),
        _load_eyeball("OBJ/eyeball_R.obj"),
    ])

    eye_num_faces = len(eye_faces)
    eye_num_verts = eye_positions.shape[0]
    eye_num_uvs = eye_uvs.shape[0]

    eye_vertex_normals = compute_vertex_normals(eye_positions, eye_faces)

    eye_tri_vertex_idx_np = np.array(
        [[f[0][0], f[1][0], f[2][0]] for f in eye_faces], dtype=np.int32)
    eye_tri_uv_idx_np = np.array(
        [[(f[0][1] if f[0][1] is not None else 0),
          (f[1][1] if f[1][1] is not None else 0),
          (f[2][1] if f[2][1] is not None else 0)] for f in eye_faces],
        dtype=np.int32)

    eyeball = build_and_upload(eye_positions, "eyeball", mesh_faces=eye_faces)

    eye_vertex_normals_field = ti.Vector.field(3, ti.f32, shape=eye_num_verts)
    eye_vertex_normals_field.from_numpy(eye_vertex_normals.astype(np.float32))

    eye_tri_vertex_idx = ti.Vector.field(3, ti.i32, shape=eye_num_faces)
    eye_tri_vertex_idx.from_numpy(eye_tri_vertex_idx_np)
    eye_tri_uv_idx = ti.Vector.field(3, ti.i32, shape=eye_num_faces)
    eye_tri_uv_idx.from_numpy(eye_tri_uv_idx_np)

    eye_uvs_field = ti.Vector.field(2, ti.f32, shape=eye_num_uvs)
    eye_uvs_field.from_numpy(eye_uvs.astype(np.float32))

    HAS_EYES = True
    elo, ehi = eye_positions.min(axis=0), eye_positions.max(axis=0)
    print(f"eyeballs: {eye_num_faces} tris, bounds {elo} .. {ehi}")
except FileNotFoundError as e:
    HAS_EYES = False
    print(f"no eyeballs loaded ({e}); rendering head only")


# ------------------------------------------------------- back compatibility
# Flat names for the outer/inner field dicts, matching the original scene.py.
positions_field = outer["positions"]
node_bbox_min = outer["bbox_min"]
node_bbox_max = outer["bbox_max"]
node_left = outer["left"]
node_right = outer["right"]
node_tri_start = outer["tri_start"]
node_tri_count = outer["tri_count"]
leaf_triangle_indices = outer["leaf_idx"]

inner_positions_field = inner["positions"]
inner_node_bbox_min = inner["bbox_min"]
inner_node_bbox_max = inner["bbox_max"]
inner_node_left = inner["left"]
inner_node_right = inner["right"]
inner_node_tri_start = inner["tri_start"]
inner_node_tri_count = inner["tri_count"]
inner_leaf_triangle_indices = inner["leaf_idx"]