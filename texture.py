import numpy as np
import taichi as ti
import taichi.math as tm
import imageio.v3 as iio

if ti.lang.impl.get_runtime().prog is None:
    ti.init(arch=ti.gpu if ti._lib.core.with_cuda() else ti.cpu)

TEX_SIZE = 8192

# f32 at 8192^2 costs 805 MB for a vec3 map and 268 MB for a scalar one;
# all four maps together is ~2.15 GB. f16 halves that and is well past
# enough precision for 8-bit source art.
TEX_DTYPE = ti.f16
_NP_DTYPE = np.float16 if TEX_DTYPE == ti.f16 else np.float32

albedo_field = ti.Vector.field(3, dtype=TEX_DTYPE, shape=(TEX_SIZE, TEX_SIZE))
normal_field = ti.Vector.field(3, dtype=TEX_DTYPE, shape=(TEX_SIZE, TEX_SIZE))
roughness_field = ti.field(dtype=TEX_DTYPE, shape=(TEX_SIZE, TEX_SIZE))
specular_field = ti.field(dtype=TEX_DTYPE, shape=(TEX_SIZE, TEX_SIZE))


# ------------------------------------------------------------------ loading
def _read(path, channels, expect_size=None):
    """Read an image to float in [0, 1], honouring the source bit depth.

    expect_size defaults to TEX_SIZE for face maps; pass None (the eye path
    does) to skip the check when the caller will resize the image itself."""
    img = iio.imread(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)

    if img.ndim == 2:
        img = img[:, :, None]
    img = img[:, :, :channels] if channels > 1 else img[:, :, 0]

    if expect_size is None:
        expect_size = TEX_SIZE
    if expect_size:
        expected = ((expect_size, expect_size, channels) if channels > 1
                    else (expect_size, expect_size))
        assert img.shape == expected, f"{path}: got {img.shape}, want {expected}"
    return img


def srgb_to_linear(c):
    """Undo the sRGB transfer curve. Colour maps are authored in sRGB; every
    light transport calculation downstream assumes linear. Skipping this
    roughly doubles mid-tone albedo, and because albedo multiplies once per
    scattering event the error compounds with depth."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_albedo(path):
    albedo_field.from_numpy(srgb_to_linear(_read(path, 3)).astype(_NP_DTYPE))


def load_roughness(path):
    roughness_field.from_numpy(_read(path, 1).astype(_NP_DTYPE))


def load_specular(path):
    specular_field.from_numpy(_read(path, 1).astype(_NP_DTYPE))


def load_normal(path, flip_green=False):
    """Tangent-space normal map, decoded from [0,1] to [-1,1].

    flip_green swaps between the two conventions: OpenGL/Blender expect +Y
    up, DirectX/most game engines expect +Y down. Getting it backwards makes
    pores and wrinkles read as bumps under lighting from the wrong side."""
    data = _read(path, 3) * 2.0 - 1.0
    if flip_green:
        data[:, :, 1] *= -1.0

    lengths = np.linalg.norm(data, axis=2, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    normal_field.from_numpy((data / lengths).astype(_NP_DTYPE))


# ----------------------------------------------------------------- sampling
@ti.func
def _texel_coords(uv):
    """Bilinear setup shared by every sampler.

    uv * N - 0.5 puts sample points at texel centres; uv * (N-1) shifts the
    whole map by half a texel. UVs are clamped, since a slightly out of range
    coordinate would otherwise index negatively and read unrelated memory."""
    x = tm.clamp(uv[0], 0.0, 1.0) * TEX_SIZE - 0.5
    y = (1.0 - tm.clamp(uv[1], 0.0, 1.0)) * TEX_SIZE - 0.5

    x0 = ti.cast(tm.floor(x), ti.i32)
    y0 = ti.cast(tm.floor(y), ti.i32)
    fx = x - x0
    fy = y - y0

    x0 = tm.clamp(x0, 0, TEX_SIZE - 1)
    y0 = tm.clamp(y0, 0, TEX_SIZE - 1)
    x1 = tm.clamp(x0 + 1, 0, TEX_SIZE - 1)
    y1 = tm.clamp(y0 + 1, 0, TEX_SIZE - 1)

    return x0, y0, x1, y1, fx, fy


@ti.func
def _bilinear3(field: ti.template(), uv):
    x0, y0, x1, y1, fx, fy = _texel_coords(uv)
    top = ti.cast(field[y0, x0], ti.f32) * (1.0 - fx) + ti.cast(field[y0, x1], ti.f32) * fx
    bot = ti.cast(field[y1, x0], ti.f32) * (1.0 - fx) + ti.cast(field[y1, x1], ti.f32) * fx
    return top * (1.0 - fy) + bot * fy


@ti.func
def sample_albedo(uv):
    return _bilinear3(albedo_field, uv)


@ti.func
def sample_roughness(uv):
    return _bilinear3(roughness_field, uv)


@ti.func
def sample_specular(uv):
    return _bilinear3(specular_field, uv)


@ti.func
def sample_normal(uv):
    """Tangent-space normal, renormalized: interpolating unit vectors
    shortens them, and an unnormalized normal biases every dot product."""
    return tm.normalize(_bilinear3(normal_field, uv))


@ti.func
def apply_normal_map(uv, N, T, handedness):
    """Perturb a shading normal by the tangent-space map.

    T comes from compute_tangents, handedness from the same call: a head
    unwrap is usually mirrored across the centre line, and without the sign
    the bitangent points backwards on one half of the face."""
    n_ts = sample_normal(uv)
    T_o = tm.normalize(T - N * tm.dot(N, T))
    B = tm.cross(N, T_o) * handedness
    return tm.normalize(n_ts[0] * T_o + n_ts[1] * B + n_ts[2] * N)