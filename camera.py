import taichi as ti 
import taichi.math as tm
import numpy as np 
from helpers import normalize


def make_camera(position, look, up, fov, width, height):
    position = np.array(position, dtype = np.float32)
    look = np.array(look, dtype=np.float32)
    up = np.array(up, dtype=np.float32)


    #camera basus uvw 
    w = normalize(position - look)
    u = normalize(np.cross(up, w))
    v = np.cross(w, u)

    return {
        "position" : position,
        "w" : w,
        "u" : u,
        "v" : v,
        "fov" : fov,
        "tan_half_fov" : float(np.tan(np.radians(fov)/2.0)),
        "aspect" : float(width)/float(height)
    }


#taichi fields holding camera state

cam_pos = ti.Vector.field(3, dtype = ti.f32, shape = ())
cam_u = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_v = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_w = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_params = ti.field(dtype=ti.f32, shape=4)  # tan_half_fov, aspect, img_width, img_height

def setup_camera(position, look, up, fov, width, height):
    cam = make_camera(position, look, up, fov, width, height)

    cam_pos[None] = cam["position"]
    cam_u[None] = cam["u"]
    cam_v[None] = cam["v"]
    cam_w[None] = cam["w"]
    cam_params[0] = cam["tan_half_fov"]
    cam_params[1] = cam["aspect"]
    cam_params[2] = float(width)
    cam_params[3] = float(height)



@ti.func
def sample_ray_through_pixel(px : ti.f32, py : ti.f32):
    tan_half_fov = cam_params[0]
    aspect = cam_params[1]
    img_width = cam_params[2]
    img_height = cam_params[3]

    jittered_px = px + ti.random(ti.f32)
    jittered_py = py + ti.random(ti.f32)

    # map btn [-1, 1] so that zero is in the middle
    ndc_x = jittered_px / img_width * 2.0 - 1.0
    ndc_y = jittered_py / img_height * 2.0 - 1.0

    screen_x = ndc_x * aspect * tan_half_fov
    screen_y = ndc_y * tan_half_fov

    direction = tm.normalize(screen_x * cam_u[None] + screen_y * cam_v[None] - cam_w[None])
    return cam_pos[None], direction 

