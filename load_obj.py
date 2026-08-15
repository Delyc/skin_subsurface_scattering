import numpy as np 

def load_obj(path):
    positions = []
    uvs = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                positions.append([parts[1], parts[2], parts[3]]) #xyz
            elif line.startswith('vt '):
                parts = line.split()
                uvs.append([parts[1], parts[2]])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    idxs = part.split('/')
                    v_idx = int(idxs[0]) - 1
                    uv_idx = int(idxs[1]) - 1 if len(idxs) > 1 and idxs[1] else - 1
                    face.append((v_idx, uv_idx))
                
                faces.append(face)
        
    return positions, uvs, faces


def triangulate_faces(faces):
    tri_faces = []
    for face in faces:
        tri_faces.append((face[0], face[1], face[2]))
        tri_faces.append((face[0], face[2], face[3]))
    return tri_faces




                    




