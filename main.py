import taichi as ti
ti.init(arch = ti.gpu)
import taichi.math as tm

from camera import setup_camera, sample_ray_through_pixel
from shading import interpolate_normal, interpolate_uv
from texture import sample_albedo, load_albedo, load_roughness, sample_roughness, load_specular, sample_specular, load_normal
from lighting import set_light
from radiance import radiance
from sss import walk_stats, walk_rgb, walk_tir
from scene import overflow_count

img_width = 400
img_height = 400

# SAMPLES_PER_PIXEL = 64

BATCH = 8              # samples per batch
NUM_BATCHES = 1024



# camera tight on the cheek
setup_camera(position=[-30, 205, 160], look=[-38, 205, 55], up=[0, 1, 0],
             fov=10.0, width=img_width, height=img_height)

# small light almost parallel to the skin, from the left
set_light(position=[-220, 195, 20], u_vec=[0, 30, 0], v_vec=[0, 0, 30],
          emission=[3000.0, 3000.0, 3000.0])





color = ti.Vector.field(3, dtype = ti.f32, shape=(img_width, img_height))
accum_buffer = ti.Vector.field(3, dtype = ti.f32, shape = (img_width, img_height))

#load textures
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

EXPOSURE = 0.05      

@ti.kernel
def finalize(total_samples: ti.i32):
    for px, py in color:
        c = accum_buffer[px, py] / total_samples
        c = c * EXPOSURE
        lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
        c = c / (1.0 + lum)
        c = tm.pow(tm.max(c, 0.0), 1.0/2.2)
        color[px, py] = tm.clamp(c, 0.0, 1.0)


import time
accum_buffer.fill(0.0)
start = time.time()

for b in range(NUM_BATCHES):
    render_batch()
    done = (b + 1) * BATCH
    finalize(done)
    img = color.to_numpy()
    ti.tools.imwrite(img, 'test.png')
    print(f"{done} spp   {time.time() - start:.1f}s", flush=True)


