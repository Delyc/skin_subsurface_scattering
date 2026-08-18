import os
import time

# Resolution has to be set before scene.py is imported: the per-pixel BVH
# traversal stack is allocated from it, and a render larger than the stack
# writes out of bounds on every ray.
IMG_WIDTH = 400
IMG_HEIGHT = 400
os.environ.setdefault("RENDER_W", str(IMG_WIDTH))
os.environ.setdefault("RENDER_H", str(IMG_HEIGHT))

import taichi as ti
ti.init(arch=ti.gpu)
import taichi.math as tm

from camera import setup_camera, sample_ray_through_pixel
from texture import (load_albedo, load_roughness, load_specular, load_normal)
from lighting import set_light
from radiance import radiance
from sss import walk_stats, walk_rgb, walk_tir, walk_diag
from scene import overflow_count, RESOLUTION

assert RESOLUTION == (IMG_WIDTH, IMG_HEIGHT), \
    f"traversal stack is {RESOLUTION}, render is {(IMG_WIDTH, IMG_HEIGHT)}"

BATCH = 8
NUM_BATCHES = 10000
SAVE_EVERY = 4          # writing a PNG every batch costs more than it helps

# EXPOSURE = 0.03

# camera tight on the cheek.
# flip_y=False because ti.tools.imwrite puts row 0 at the BOTTOM; the default
# assumes row 0 is the top, which renders the head upside down.
# setup_camera(position=[-30, 205, 160], look=[-38, 205, 55], up=[0, 1, 0],
#              fov=10.0, width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)

# # small light almost parallel to the skin, from the left
# set_light(position=[-220, 195, 20], u_vec=[0, 30, 0], v_vec=[0, 0, 30],
#           emission=[3000.0, 3000.0, 3000.0])


#goodd
# setup_camera(position=[0, 190, 420], look=[0, 175, 0], up=[0, 1, 0],
#              fov=32.0, width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)
# set_light(position=[-90, 340, 380], u_vec=[70, 0, 0], v_vec=[0, 0, 70],
#           emission=[9000.0, 9000.0, 9000.0])





# set_light(position=[-200, 360, 300], u_vec=[70, 0, 0], v_vec=[0, 0, 70],
#           emission=[9000.0, 9000.0, 9000.0])

# ear lit
# camera outside the left ear, looking inward
# setup_camera(position=[-380, 200, 60], look=[-80, 195, 15], up=[0, 1, 0],
#              fov=20.0, width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)

# set_light(position=[-105, 190, -240], u_vec=[50, 0, 0], v_vec=[0, 50, 0],
#           emission=[8000.0, 8000.0, 8000.0])


# setup_camera(position=[-300, 190, 90], look=[-52, 190, 13], up=[0, 1, 0],
#              fov=16.0, width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)
# set_light(position=[-52, 195, -30], u_vec=[0, 35, 0], v_vec=[0, 0, 35],
#           emission=[20000.0, 20000.0, 20000.0])
# EXPOSURE = 0.05

# setup_camera(position=[-292.7, 159.1, 65.9], look=[-49.8, 188.5, 14.1], up=[0, 1, 0],
#              fov=15.0, width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)

# set_light(position=[-10, 192, 10], u_vec=[0, 70, 0], v_vec=[0, 0, 70],
#           emission=[9000.0, 9000.0, 9000.0])
# EXPOSURE = 0.15


#test
setup_camera(position=[-140, 205, 180], look=[-49.8, 188.5, 14.1], up=[0, 1, 0],
             fov=13.0, width=IMG_WIDTH, height=IMG_HEIGHT, flip_y=False)

# behind the ear from THIS camera = deeper into the head, toward +x/-z
set_light(position=[-95, 190, -25], u_vec=[0, 12, 0], v_vec=[0, 0, 12],
          emission=[5000.0, 5000.0, 5000.0])
EXPOSURE = 0.07

color = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_WIDTH, IMG_HEIGHT))
accum_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_WIDTH, IMG_HEIGHT))

load_albedo("TGA/Face/Face_Albedo.tga")
load_roughness("TGA/Face/Face_Roughness.tga")
load_specular("TGA/Face/Face_Specular.tga")
load_normal("TGA/Face/Face_Normal.tga")


@ti.kernel
def render_batch():
    for px, py in accum_buffer:
        for s in range(BATCH):
            ray_origin, ray_dir = sample_ray_through_pixel(float(px), float(py))
            accum_buffer[px, py] += radiance(ray_origin, ray_dir, px, py)


@ti.func
def linear_to_srgb(c):
    """Matches the decode in textures.py. Plain 2.2 gamma is close but not
    the same curve, and the mismatch shows in the darkest tones."""
    out = tm.vec3(0.0, 0.0, 0.0)
    for i in ti.static(range(3)):
        x = tm.max(c[i], 0.0)
        if x <= 0.0031308:
            out[i] = x * 12.92
        else:
            out[i] = 1.055 * tm.pow(x, 1.0 / 2.4) - 0.055
    return out


@ti.kernel
def finalize(total_samples: ti.i32):
    for px, py in color:
        c = accum_buffer[px, py] / total_samples
        c = c * EXPOSURE
        lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        c = c / (1.0 + lum)
        color[px, py] = tm.clamp(linear_to_srgb(tm.max(c, 0.0)), 0.0, 1.0)


def report():
    """The walk statistics are the only way to see energy quietly vanishing.

    walk_stats[3] is the one that matters: those walks hit MAX_WALK_STEPS and
    their energy is simply dropped. Roulette kills (index 1) are unbiased and
    cost nothing but noise."""
    escaped, roulette, lost, capped = (int(walk_stats[i]) for i in range(4))
    total = max(escaped + roulette + lost + capped, 1)

    print(f"  walks: {escaped} escaped ({100*escaped/total:.1f}%)  "
          f"{roulette} roulette  {lost} lost-geometry  "
          f"{capped} step-capped ({100*capped/total:.2f}% <- real energy loss)")

    if escaped:
        rgb = [walk_rgb[i] / escaped for i in range(3)]
        print(f"  mean exit throughput RGB: "
              f"({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})")

    walks = max(total, 1)
    print(f"  per walk: {walk_diag[0]/walks:.1f} scatters  "
          f"{walk_diag[1]/walks:.1f} wall-crossings  "
          f"{walk_diag[3]/walks:.1f} steps  "
          f"{100*walk_diag[2]/max(walk_diag[3],1):.1f}% in epidermis")
    print(f"  TIR bounces: {int(walk_tir[None])}   "
          f"BVH stack overflows: {int(overflow_count[None])}")

    if capped > 0.02 * total:
        print("  WARNING: over 2% of walks hit the step cap; that energy is "
              "gone. Raise MAX_WALK_STEPS or add Dwivedi guiding.")
    if int(overflow_count[None]) > 0:
        print("  WARNING: BVH stack overflowed - triangles were skipped, "
              "which shows up as holes. Raise STACK_MAX.")


accum_buffer.fill(0.0)
start = time.time()

for b in range(NUM_BATCHES):
    render_batch()
    done = (b + 1) * BATCH

    if (b + 1) % SAVE_EVERY == 0 or b == NUM_BATCHES - 1:
        finalize(done)
        ti.tools.imwrite(color.to_numpy(), "300.png")
        print(f"{done} spp   {time.time() - start:.1f}s", flush=True)
        report()