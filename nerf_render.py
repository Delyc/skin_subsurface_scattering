import os
import json
import time

import numpy as np

IMG_WIDTH = 400
IMG_HEIGHT = 400
os.environ.setdefault("RENDER_W", str(IMG_WIDTH))
os.environ.setdefault("RENDER_H", str(IMG_HEIGHT))

import taichi as ti
ti.init(arch=ti.gpu)
import taichi.math as tm

from camera import setup_camera, sample_ray_through_pixel
from texture import load_albedo, load_roughness, load_specular, load_normal
from lighting import set_lights
from radiance import radiance, primary_aov
from scene import RESOLUTION, HAS_EYES
from denoise import denoise

assert RESOLUTION == (IMG_WIDTH, IMG_HEIGHT)

# ----------------------------------------------------------------- settings
NUM_VIEWS = 100
BATCH = 8
NUM_BATCHES = 132          # 1056 spp
EXPOSURE = 0.02
FOV = 32.0

OUT_DIR = "nerf"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)


CENTER = np.array([0.0, 150.0, 25.0], dtype=np.float64)
RADIUS = 420.0


_KEY = [1000.0, 1000.0, 1000.0]
set_lights([
    {"position": [-300, 205, 160], "u_vec": [0, 130, 0], "v_vec": [0, 0, 130],
     "emission": _KEY},
    {"position": [300, 205, 160], "u_vec": [0, 130, 0], "v_vec": [0, 0, 130],
     "emission": _KEY},
    {"position": [0, 205, -320], "u_vec": [130, 0, 0], "v_vec": [0, 130, 0],
     "emission": _KEY},
    {"position": [0, 460, 140], "u_vec": [130, 0, 0], "v_vec": [0, 0, 130],
     "emission": _KEY},
])

# ---------------------------------------------------------------- textures
load_albedo("TGA/Face/Face_Albedo.tga")
load_roughness("TGA/Face/Face_Roughness.tga")
load_specular("TGA/Face/Face_Specular.tga")
load_normal("TGA/Face/Face_Normal.tga")

if HAS_EYES:
    from eye import load_eye_textures
    load_eye_textures(
        "TGA/Eyes/Eyes_Balls_Diffuse.tga",
        "TGA/Eyes/Eyes_Balls_Normals.tga",
        "TGA/Eyes/Eyes_Balls_Roughness.tga",
        "TGA/Eyes/Eyes_Balls_Spec.tga")

# ------------------------------------------------------------------ buffers
accum_buffer = ti.Vector.field(3, ti.f32, shape=(IMG_WIDTH, IMG_HEIGHT))
albedo_buffer = ti.Vector.field(3, ti.f32, shape=(IMG_WIDTH, IMG_HEIGHT))
normal_buffer = ti.Vector.field(3, ti.f32, shape=(IMG_WIDTH, IMG_HEIGHT))
linear_buffer = ti.Vector.field(3, ti.f32, shape=(IMG_WIDTH, IMG_HEIGHT))


@ti.kernel
def render_batch():
    for px, py in accum_buffer:
        for s in range(BATCH):
            o, d = sample_ray_through_pixel(float(px), float(py))
            accum_buffer[px, py] += radiance(o, d, px, py)


@ti.kernel
def fill_aov():
    for px, py in albedo_buffer:
        o, d = sample_ray_through_pixel(float(px), float(py), 0)
        alb, nrm = primary_aov(o, d, px, py)
        albedo_buffer[px, py] = alb
        normal_buffer[px, py] = nrm


@ti.kernel
def extract_linear(total: ti.i32):
    for px, py in linear_buffer:
        linear_buffer[px, py] = accum_buffer[px, py] / total * EXPOSURE


def tonemap_np(lin):
    lum = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1]
           + 0.0722 * lin[..., 2])[..., None]
    c = np.maximum(lin / (1.0 + lum), 0.0)
    srgb = np.where(c <= 0.0031308, c * 12.92,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


# --------------------------------------------------- camera pose helpers
def fibonacci_sphere(n):
    """n roughly-even points on a unit sphere (spiral). Even coverage of all
    azimuths and elevations, including the back and top, so the trained NeRF
    can be viewed from anywhere."""
    pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))         # golden angle
    for i in range(n):
        y = 1.0 - (i / (n - 1)) * 2.0          # 1 -> -1
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts.append([np.cos(theta) * r, y, np.sin(theta) * r])
    return np.array(pts)


def camera_to_world(eye, center, up=np.array([0.0, 1.0, 0.0])):
    """4x4 camera-to-world in the NeRF/OpenGL convention: camera looks down
    its local -z, +x right, +y up. Matches the ray generation in a standard
    from-scratch NeRF, so poses drop in without a coordinate flip."""
    fwd = center - eye
    fwd = fwd / np.linalg.norm(fwd)
    # guard against up being parallel to fwd (straight up/down views)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = true_up
    m[:3, 2] = -fwd          # camera looks down -z
    m[:3, 3] = eye
    return m


# ----------------------------------------------------------------- render
def render_one():
    accum_buffer.fill(0.0)
    fill_aov()
    albedo_np = albedo_buffer.to_numpy()
    normal_np = normal_buffer.to_numpy()

    for _ in range(NUM_BATCHES):
        render_batch()

    extract_linear(NUM_BATCHES * BATCH)
    lin = linear_buffer.to_numpy()
    clean = denoise(lin, albedo=albedo_np, normal=normal_np, hdr=True)
    return tonemap_np(clean)


def main():
    dirs = fibonacci_sphere(NUM_VIEWS)
    frames = []
    start = time.time()

    for i in range(NUM_VIEWS):
        eye = CENTER + dirs[i] * RADIUS
        view_dir = (CENTER - eye) / np.linalg.norm(CENTER - eye)
        up = [0, 1, 0]
        if abs(view_dir[1]) > 0.99:      # near the poles, use a different up
            up = [0, 0, 1]
        setup_camera(position=eye.tolist(), look=CENTER.tolist(),
                     up=up, fov=FOV,
                     width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)

        img = render_one()
        name = f"r_{i:03d}.png"
        ti.tools.imwrite(img, os.path.join(IMG_DIR, name))

        c2w = camera_to_world(eye, CENTER)
        frames.append({
            "file_path": f"images/{name}",
            "transform_matrix": c2w.tolist(),
        })

        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (NUM_VIEWS - i - 1)
        print(f"[{i+1}/{NUM_VIEWS}] {name}  "
              f"{elapsed/60:.1f} min elapsed, ~{eta/60:.0f} min left",
              flush=True)

    # camera angle -> focal length in the transforms.json convention
    camera_angle_x = 2.0 * np.arctan(
        np.tan(np.radians(FOV) / 2.0) * (IMG_WIDTH / IMG_HEIGHT))

    meta = {
        "camera_angle_x": float(camera_angle_x),
        "w": IMG_WIDTH,
        "h": IMG_HEIGHT,
        "cx": IMG_WIDTH / 2.0,
        "cy": IMG_HEIGHT / 2.0,
        "aabb_scale": 1,
        "frames": frames,
    }
    with open(os.path.join(OUT_DIR, "transforms.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"done: {NUM_VIEWS} views in {(time.time()-start)/60:.1f} min")
    print(f"wrote {OUT_DIR}/transforms.json")


if __name__ == "__main__":
    main()