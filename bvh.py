import numpy as np 
from load_obj import load_obj
from load_obj import triangulate_faces

#each triangles min and max
def compute_triangle_bboxes(positions, faces):
    bboxes = []
    for face in faces:
        v_idx0 = face[0][0]
        v_idx1 = face[1][0]
        v_idx2 = face[2][0]

        p0 = positions[v_idx0]
        p1 = positions[v_idx1]
        p2 = positions[v_idx2]

        min_corner = np.min(np.array([p0, p1, p2]), axis = 0)
        max_corner = np.max(np.array([p0, p1, p2]), axis = 0)

        bboxes.append((min_corner, max_corner))
    
    return bboxes


def compute_root_bbox(bboxes):
    all_mins = np.array([b[0] for b in bboxes])
    all_maxs = np.array([b[1] for b in bboxes])

    root_min = np.min(all_mins, axis = 0)
    root_max = np.max(all_maxs, axis = 0)

    return root_min, root_max


def centroid(p0, p1, p2):
    return (p0 + p1 + p2) / 3


def compute_centroids(positions, faces):
    centroids = []
    for face in faces:
        v_idx0 = face[0][0]
        v_idx1 = face[1][0]
        v_idx2 = face[2][0]

        p0 = positions[v_idx0]
        p1 = positions[v_idx1]
        p2 = positions[v_idx2]

        centroids.append(centroid(p0, p1, p2))

    return np.array(centroids)


def longest_axis(min_corner, max_corner):
    extent = max_corner - min_corner
    return np.argmax(extent)

def split_triangles(centroids, axis):
    values = centroids[:, axis]
    median = np.median(values)

    left_indices = np.where(values <= median)[0]
    right_indices = np.where(values > median)[0]

    return left_indices, right_indices


class BVHNode:
    def __init__(self, bbox_min, bbox_max, left = None, right = None, triangle_indices = None):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.left = left
        self.right = right
        self.triangle_indices = triangle_indices


LEAF_SIZE = 4

def build_bvh(positions, faces, centroids, triangle_indices):
    sub_faces = [faces[i] for i in triangle_indices]
    bboxes = compute_triangle_bboxes(positions, sub_faces)
    bbox_min, bbox_max = compute_root_bbox(bboxes)

    if len(triangle_indices) <= LEAF_SIZE:
        return BVHNode(bbox_min, bbox_max, triangle_indices = triangle_indices)
    
    sub_centroids = centroids[triangle_indices]
    axis = longest_axis(bbox_min, bbox_max)
    left_local, right_local = split_triangles(sub_centroids, axis)

    left_indices = triangle_indices[left_local]
    right_indices = triangle_indices[right_local]

    left_node = build_bvh(positions, faces, centroids, left_indices)
    right_node = build_bvh(positions, faces, centroids, right_indices)

    return BVHNode(bbox_min, bbox_max, left = left_node, right = right_node)




def flatten_bvh(root):
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
            start = len(leaf_triangle_indices)
            leaf_triangle_indices.extend(node.triangle_indices.tolist())
            node_tri_start[my_id] = start
            node_tri_count[my_id] = len(node.triangle_indices)
        
        else:
            left_id = visit(node.left)
            right_id = visit(node.right)
            node_left[my_id] = left_id
            node_right[my_id] = right_id

        return my_id
    visit(root)

    return(
        np.array(node_bbox_min, dtype=np.float32),
        np.array(node_bbox_max, dtype=np.float32),
        np.array(node_left, dtype=np.int32),
        np.array(node_right, dtype=np.int32),
        np.array(node_tri_start, dtype=np.int32),
        np.array(node_tri_count, dtype=np.int32),
        np.array(leaf_triangle_indices, dtype=np.int32),
    )