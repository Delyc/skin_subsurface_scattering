import numpy as np
import taichi as ti
import taichi.math as tm


def normalize(vector):
    return vector / np.linalg.norm(vector)


@ti.func
def environment_light(direction):
    return tm.vec3(0.05, 0.05, 0.05)
    # return tm.vec3(0.0, 0.0, 0.0)


def compute_face_normals(faces, positions):
    face_normals = []

    for face in faces:
        v_idx0 = face[0][0]
        v_idx1 = face[1][0]
        v_idx2 = face[2][0]

        p0 = positions[v_idx0]
        p1 = positions[v_idx1]
        p2 = positions[v_idx2]

        edge1 = p1 - p0
        edge2 = p2 - p0

        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        face_normals.append(normal)

    return np.array(face_normals)


def compute_vertex_normal(faces, positions, face_normals):
    vertex_normals = np.zeros_like(positions)

    for face, fn in zip(faces, face_normals):
        for (v_idx, uv_idx) in face:
            vertex_normals[v_idx] += fn

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vertex_normals = vertex_normals / norms

    return vertex_normals


def compute_tangents(positions, uvs, faces):
    """For each triangle, the 3D direction the texture's U axis points.
    Solves e = duv.u*T + duv.v*B for two edges (Lengyel's method)."""
    tangents = np.zeros((len(faces), 3), dtype=np.float32)
    uvs_np = np.array(uvs, dtype=np.float32)
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for i, face in enumerate(faces):
        v0, v1, v2 = face[0][0], face[1][0], face[2][0]
        t0, t1, t2 = face[0][1], face[1][1], face[2][1]

        e1 = positions[v1] - positions[v0]
        e2 = positions[v2] - positions[v0]
        duv1 = uvs_np[t1] - uvs_np[t0]
        duv2 = uvs_np[t2] - uvs_np[t0]

        denom = duv1[0] * duv2[1] - duv2[0] * duv1[1]
        if abs(denom) < 1e-12:
            tangents[i] = fallback
            continue

        r = 1.0 / denom
        t = (e1 * duv2[1] - e2 * duv1[1]) * r
        n = np.linalg.norm(t)
        tangents[i] = t / n if n > 1e-12 else fallback

    return tangents


def find_boundary_edges(faces):
    """Edges used by exactly one triangle - the rim of a hole.
    Interior edges are shared by two."""
    edge_count = {}

    for face in faces:
        v0, v1, v2 = face[0][0], face[1][0], face[2][0]
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (min(a, b), max(a, b))     # (5,9) and (9,5) are one edge
            edge_count[key] = edge_count.get(key, 0) + 1

    return [e for e, c in edge_count.items() if c == 1]


def boundary_center(positions, boundary):
    verts = set()
    for a, b in boundary:
        verts.add(a)
        verts.add(b)
    verts = list(verts)
    return np.mean(positions[verts], axis=0), verts


def cap_boundary(positions, uvs, faces, boundary, center):

    positions = np.vstack([positions, center.astype(np.float32)])
    center_idx = len(positions) - 1


    uvs = list(uvs) + [[0.0, 0.0]]
    center_uv_idx = len(uvs) - 1

    uv_of_vertex = {}
    for face in faces:
        for (v_idx, uv_idx) in face:
            if v_idx not in uv_of_vertex:
                uv_of_vertex[v_idx] = uv_idx

    new_faces = list(faces)
    for a, b in boundary:
        new_faces.append((
            (a, uv_of_vertex.get(a, center_uv_idx)),
            (b, uv_of_vertex.get(b, center_uv_idx)),
            (center_idx, center_uv_idx),
        ))

    return positions, uvs, new_faces