import numpy as np
import os, glob, ctypes
_tbb = glob.glob(os.path.join(os.path.dirname(__file__), ".venv/**/libtbb.so.12"), recursive=True)
if _tbb:
    ctypes.CDLL(_tbb[0])
import oidn


def denoise(color, albedo=None, normal=None, hdr=True):
    """Denoise a rendered image with Intel Open Image Denoise.

    color, albedo, normal are (H, W, 3) float32 arrays. albedo and normal are
    the noise-free guide buffers from primary_aov: passing them keeps real
    detail (pores, iris, edges) that a colour-only denoise would smear. hdr
    should be True for un-tonemapped linear radiance, which is what the
    accumulation buffer holds - denoise before tonemapping, not after."""
    color = np.ascontiguousarray(color, dtype=np.float32)
    h, w, _ = color.shape
    out = np.zeros_like(color)

    device = oidn.NewDevice()
    oidn.CommitDevice(device)

    flt = oidn.NewFilter(device, "RT")

    oidn.SetSharedFilterImage(
        flt, "color", color, oidn.FORMAT_FLOAT3, w, h)

    if albedo is not None:
        albedo = np.ascontiguousarray(albedo, dtype=np.float32)
        oidn.SetSharedFilterImage(
            flt, "albedo", albedo, oidn.FORMAT_FLOAT3, w, h)

    if normal is not None:
        normal = np.ascontiguousarray(normal, dtype=np.float32)
        oidn.SetSharedFilterImage(
            flt, "normal", normal, oidn.FORMAT_FLOAT3, w, h)

    oidn.SetSharedFilterImage(
        flt, "output", out, oidn.FORMAT_FLOAT3, w, h)

    # 'hdr' tells OIDN the colour is unbounded linear radiance rather than a
    # [0,1] tonemapped image, so it doesn't clip highlights.
    try:
        oidn.SetFilter1b(flt, "hdr", hdr)
    except AttributeError:
        pass  # older binding: hdr inferred from data

    oidn.CommitFilter(flt)
    oidn.ExecuteFilter(flt)

    err = oidn.GetDeviceError(device)
    oidn.ReleaseFilter(flt)
    oidn.ReleaseDevice(device)

    if isinstance(err, tuple) and err[0] != 0:
        raise RuntimeError(f"OIDN error: {err}")

    return out