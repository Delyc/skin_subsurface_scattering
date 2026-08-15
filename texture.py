import taichi as ti 
import taichi.math as tm 
import numpy as np
import imageio.v3 as iio


TEX_SIZE = 8192

#albedo
albedo_field = ti.Vector.field(3, dtype = ti.f32, shape = (TEX_SIZE, TEX_SIZE))

def load_albedo(path):
    img = iio.imread(path)
    linear = img.astype(np.float32) / 255.0
    assert linear.shape == (TEX_SIZE, TEX_SIZE, 3), f"got {linear.shape}"
    albedo_field.from_numpy(linear)

@ti.func
def sample_albedo(uv):
    x = uv[0] * (TEX_SIZE - 1)
    y = (1.0 - uv[1]) * (TEX_SIZE - 1)

    x0 = int(tm.floor(x))
    y0 = int(tm.floor(y))
    x1 = tm.min(x0 + 1, TEX_SIZE - 1)
    y1 = tm.min(y0 + 1, TEX_SIZE - 1)

    fx = x - x0
    fy = y - y0

    c00 = albedo_field[y0, x0]
    c10 = albedo_field[y0, x1]
    c01 = albedo_field[y1, x0]
    c11 = albedo_field[y1, x1]

    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx

    return top * (1.0 - fy) + bot * fy




#roughness
roughness_field = ti.field(dtype=ti.f32, shape=(TEX_SIZE, TEX_SIZE))

def load_roughness(path):
    img = iio.imread(path)
    gray = img[:, :, 0].astype(np.float32) / 255.0
    assert gray.shape == (TEX_SIZE, TEX_SIZE), f"got {gray.shape}"
    roughness_field.from_numpy(gray)


@ti.func
def sample_roughness(uv):
    x = uv[0] * (TEX_SIZE - 1)
    y = (1.0 - uv[1]) * (TEX_SIZE - 1)

    x0 = int(tm.floor(x))
    y0 = int(tm.floor(y))
    x1 = tm.min(x0 + 1, TEX_SIZE - 1)
    y1 = tm.min(y0 + 1, TEX_SIZE - 1)

    fx = x - x0
    fy = y - y0

    c00 = roughness_field[y0, x0]
    c10 = roughness_field[y0, x1]
    c01 = roughness_field[y1, x0]
    c11 = roughness_field[y1, x1]

    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx

    return top * (1.0 - fy) + bot * fy



#specular
specular_field = ti.field(dtype=ti.f32, shape=(TEX_SIZE, TEX_SIZE))


def load_specular(path):
    img = iio.imread(path)
    gray = img[:, :, 0].astype(np.float32) / 255.0
    assert gray.shape == (TEX_SIZE, TEX_SIZE), f"got {gray.shape}"
    specular_field.from_numpy(gray)


@ti.func
def sample_specular(uv):
    x = uv[0] * (TEX_SIZE - 1)
    y = (1.0 - uv[1]) * (TEX_SIZE - 1)

    x0 = int(tm.floor(x))
    y0 = int(tm.floor(y))
    x1 = tm.min(x0 + 1, TEX_SIZE - 1)
    y1 = tm.min(y0 + 1, TEX_SIZE - 1)

    fx = x - x0
    fy = y - y0

    c00 = specular_field[y0, x0]
    c10 = specular_field[y0, x1]
    c01 = specular_field[y1, x0]
    c11 = specular_field[y1, x1]

    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx

    return top * (1.0 - fy) + bot * fy
    
    

#normal 
normal_field = ti.Vector.field(3, dtype=ti.f32, shape=(TEX_SIZE, TEX_SIZE))


def load_normal(path):
    img = iio.imread(path)
    data = img.astype(np.float32) / 255.0
    assert data.shape == (TEX_SIZE, TEX_SIZE, 3), f"got {data.shape}"
    normal_field.from_numpy(data)


@ti.func
def sample_normal(uv):
    x = uv[0] * (TEX_SIZE - 1)
    y = (1.0 - uv[1]) * (TEX_SIZE - 1)

    x0 = int(tm.floor(x))
    y0 = int(tm.floor(y))
    x1 = tm.min(x0 + 1, TEX_SIZE - 1)
    y1 = tm.min(y0 + 1, TEX_SIZE - 1)

    fx = x - x0
    fy = y - y0

    c00 = normal_field[y0, x0]
    c10 = normal_field[y0, x1]
    c01 = normal_field[y1, x0]
    c11 = normal_field[y1, x1]

    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx

    return top * (1.0 - fy) + bot * fy 
