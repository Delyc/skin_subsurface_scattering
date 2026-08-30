import numpy as np

from load_obj import load_obj, triangulate_faces


LEAF_SIZE = 4
BBOX_PAD = 1e-6  # flat slabs -> 0 * inf -> NaN in slab tests


def compute_triangle_bboxes(positions, faces):
    """Per-triangle min/max corners. Returns two (N, 3) arrays."""
    assert all(len(f) == 3 for f in faces), "there is a quad"
    idx = np.array([[f[0][0], f[1][0], f[2][0]] for f in faces])
    tris = positions[idx]  # (N, 3, 3)
    return tris.min(axis=1), tris.max(axis=1)


def compute_centroids(positions, faces):
    """Per-triangle centroid. Returns an (N, 3) array."""
    assert all(len(f) == 3 for f in faces), "there is a quad"
    idx = np.array([[f[0][0], f[1][0], f[2][0]] for f in faces])
    return positions[idx].mean(axis=1)


def longest_axis(min_corner, max_corner):
    return int(np.argmax(max_corner - min_corner))


def split_triangles(centroids, axis):
    """Median split by sorted rank, so both sides are always non-empty."""
    order = np.argsort(centroids[:, axis], kind="stable")
    mid = len(order) // 2
    return order[:mid], order[mid:]


class BVHNode:
    def __init__(self, bbox_min, bbox_max, left=None, right=None,
                 triangle_indices=None):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.left = left
        self.right = right
        self.triangle_indices = triangle_indices


def build_bvh(tri_min, tri_max, centroids, triangle_indices):
    bbox_min = tri_min[triangle_indices].min(axis=0) - BBOX_PAD
    bbox_max = tri_max[triangle_indices].max(axis=0) + BBOX_PAD

    if len(triangle_indices) <= LEAF_SIZE:
        return BVHNode(bbox_min, bbox_max, triangle_indices=triangle_indices)

    axis = longest_axis(bbox_min, bbox_max)
    left_local, right_local = split_triangles(centroids[triangle_indices], axis)

    return BVHNode(
        bbox_min,
        bbox_max,
        left=build_bvh(tri_min, tri_max, centroids,
                       triangle_indices[left_local]),
        right=build_bvh(tri_min, tri_max, centroids,
                        triangle_indices[right_local]),
    )


def flatten_bvh(root):
    """Depth-first flatten into flat arrays ready for Taichi fields."""
    node_bbox_min = []
    node_bbox_max = []
    node_left = []
    node_right = []
    node_tri_start = []
    node_tri_count = []
    leaf_triangle_indices = []

    def visit(node):
        my_id = len(node_bbox_min)
        node_bbox_min.append(node.bbox_min)
        node_bbox_max.append(node.bbox_max)
        node_left.append(-1)
        node_right.append(-1)
        node_tri_start.append(-1)
        node_tri_count.append(0)

        if node.triangle_indices is not None:
            node_tri_start[my_id] = len(leaf_triangle_indices)
            node_tri_count[my_id] = len(node.triangle_indices)
            leaf_triangle_indices.extend(node.triangle_indices.tolist())
        else:
            node_left[my_id] = visit(node.left)
            node_right[my_id] = visit(node.right)

        return my_id

    visit(root)

    return (
        np.array(node_bbox_min, dtype=np.float32),
        np.array(node_bbox_max, dtype=np.float32),
        np.array(node_left, dtype=np.int32),
        np.array(node_right, dtype=np.int32),
        np.array(node_tri_start, dtype=np.int32),
        np.array(node_tri_count, dtype=np.int32),
        np.array(leaf_triangle_indices, dtype=np.int32),
    )


def build_from_obj(path):
    """Load an OBJ and return (positions, uvs, tris, flattened BVH)."""
    positions, uvs, faces = load_obj(path)
    tris = triangulate_faces(faces)

    tri_min, tri_max = compute_triangle_bboxes(positions, tris)
    centroids = compute_centroids(positions, tris)

    root = build_bvh(tri_min, tri_max, centroids, np.arange(len(tris)))
    return positions, uvs, tris, flatten_bvh(root)


if __name__ == "__main__":
    positions, uvs, tris, flat = build_from_obj("head.obj")
    node_min, node_max, left, right, tri_start, tri_count, leaf_idx = flat

    print(f"verts {positions.shape}")
    print(f"uvs {uvs.shape}")
    print(f"tris {len(tris)}")
    print(f"nodes {len(node_min)}  (leaves {(tri_count > 0).sum()})")
    print(f"leaf idx {len(leaf_idx)}  (must equal tris)")
    print(f"bbox {node_min[0]}  {node_max[0]}")