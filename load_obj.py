import numpy as np

def load_obj(path):
    positions, uvs, faces = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                positions.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vt':
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == 'f':
                face = []
                for p in parts[1:]:
                    i = p.split('/')
                    v = int(i[0]) - 1
                    vt = int(i[1]) - 1 if len(i) > 1 and i[1] else None
                    face.append((v, vt))
                faces.append(face)
    return np.array(positions, np.float32), np.array(uvs, np.float32), faces

def triangulate_faces(faces):
    return [(f[0], f[i], f[i+1]) for f in faces for i in range(1, len(f)-1)]