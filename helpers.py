import numpy as np
import taichi as ti
import taichi.math as tm


def normalize(vector):
    n = np.linalg.norm(vector)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / n


@ti.func
def environment_light(direction):
    return tm.vec3(0.0, 0.0, 0.0)


def _tri_indices(faces):
    assert all(len(f) == 3 for f in faces), "there is a quad"
    return np.array([[f[0][0], f[1][0], f[2][0]] for f in faces])


def compute_face_normals(positions, faces, normalized=True):
    idx = _tri_indices(faces)
    p = positions[idx]                                  # (N, 3, 3)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # (N, 3)

    if normalized:
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        lengths[lengths < 1e-20] = 1.0
        n = n / lengths
    return n.astype(np.float32)


def compute_vertex_normals(positions, faces):
    idx = _tri_indices(faces)
    weighted = compute_face_normals(positions, faces, normalized=False)

    vn = np.zeros_like(positions, dtype=np.float32)
    for c in range(3):
        np.add.at(vn, idx[:, c], weighted)

    lengths = np.linalg.norm(vn, axis=1, keepdims=True)
    lengths[lengths < 1e-20] = 1.0
    return vn / lengths


def compute_tangents(positions, uvs, faces, vertex_normals=None):
    n_faces = len(faces)
    tangents = np.zeros((n_faces, 3), dtype=np.float32)
    handedness = np.ones(n_faces, dtype=np.float32)
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    face_normals = compute_face_normals(positions, faces)

    for i, face in enumerate(faces):
        v0, v1, v2 = face[0][0], face[1][0], face[2][0]
        t0, t1, t2 = face[0][1], face[1][1], face[2][1]

        if t0 is None or t1 is None or t2 is None:
            tangents[i] = fallback
            continue

        e1 = positions[v1] - positions[v0]
        e2 = positions[v2] - positions[v0]
        duv1 = uvs[t1] - uvs[t0]
        duv2 = uvs[t2] - uvs[t0]

        denom = duv1[0] * duv2[1] - duv2[0] * duv1[1]
        if abs(denom) < 1e-12:
            tangents[i] = fallback
            continue

        r = 1.0 / denom
        t = (e1 * duv2[1] - e2 * duv1[1]) * r
        b = (e2 * duv1[0] - e1 * duv2[0]) * r

        n = vertex_normals[v0] if vertex_normals is not None else face_normals[i]

        # Gram-Schmidt: strip the component along the normal
        t = t - n * np.dot(n, t)
        length = np.linalg.norm(t)
        tangents[i] = t / length if length > 1e-12 else fallback

        if np.dot(np.cross(n, t), b) < 0.0:
            handedness[i] = -1.0

    return tangents, handedness


def find_boundary_edges(faces):
    assert all(len(f) == 3 for f in faces), "triangulate first"
    count = {}
    directed = {}

    for face in faces:
        v0, v1, v2 = face[0][0], face[1][0], face[2][0]
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
            directed.setdefault(key, (a, b))

    return [directed[e] for e, c in count.items() if c == 1]


def boundary_loops(boundary):
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(a, []).append((a, b))

    unvisited = set(boundary)
    loops = []

    while unvisited:
        start = next(iter(unvisited))
        loop = [start]
        unvisited.discard(start)
        current = start[1]

        while True:
            following = [e for e in adjacency.get(current, []) if e in unvisited]
            if not following:
                break
            edge = following[0]
            loop.append(edge)
            unvisited.discard(edge)
            current = edge[1]
            if current == start[0]:
                break

        loops.append(loop)

    return loops


def boundary_center(positions, boundary):
    verts = sorted({v for edge in boundary for v in edge})
    return np.mean(positions[verts], axis=0), verts


def cap_boundary(positions, uvs, faces, boundary, center):
    positions = np.vstack([positions, center.astype(np.float32)])
    center_idx = len(positions) - 1

    uvs = np.vstack([uvs, np.zeros((1, 2), dtype=np.float32)])
    center_uv = len(uvs) - 1

    uv_of_vertex = {}
    for face in faces:
        for (v_idx, uv_idx) in face:
            if uv_idx is not None:
                uv_of_vertex.setdefault(v_idx, uv_idx)

    new_faces = list(faces)
    for a, b in boundary:
        new_faces.append((
            (b, uv_of_vertex.get(b, center_uv)),
            (a, uv_of_vertex.get(a, center_uv)),
            (center_idx, center_uv),
        ))

    return positions, uvs, new_faces


def close_mesh(positions, uvs, faces, verbose=True):
    boundary = find_boundary_edges(faces)
    loops = boundary_loops(boundary)

    for loop in loops:
        center, _ = boundary_center(positions, loop)
        positions, uvs, faces = cap_boundary(
            positions, uvs, faces, loop, center)

    if verbose:
        sizes = ", ".join(str(len(l)) for l in loops) or "none"
        print(f"capped {len(loops)} hole(s) [edges: {sizes}]; "
              f"{len(faces)} tris, {len(positions)} verts")

    if not is_closed(faces):
        raise RuntimeError("mesh still open after capping; likely non-manifold")
    return positions, uvs, faces


def is_closed(faces):
    return len(find_boundary_edges(faces)) == 0


def signed_volume(positions, faces):
    idx = _tri_indices(faces)
    p = positions[idx]
    return float(np.einsum("ij,ij->i",
                           p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


# ------------------------------------------------------- back compatibility
# def compute_vertex_normal(faces, positions, face_normals=None):
#     """Deprecated alias. Old order was (faces, positions, face_normals)."""
#     return compute_vertex_normals(positions, faces)


# _compute_face_normals_new = compute_face_normals


# def compute_face_normals(a, b, normalized=True):
#     """Accepts either argument order; positions is the float array."""
#     if isinstance(a, np.ndarray) and a.dtype != object:
#         positions, faces = a, b
#     else:
#         faces, positions = a, b
#     return _compute_face_normals_new(positions, faces, normalized)