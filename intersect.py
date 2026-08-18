import taichi as ti
import taichi.math as tm


T_MAX = 1e30

RAY_EPS = 1e-4


@ti.func
def safe_inv_dir(ray_dir):
    inv = tm.vec3(0.0, 0.0, 0.0)
    for i in ti.static(range(3)):
        d = ray_dir[i]
        if abs(d) > 1e-20:
            inv[i] = 1.0 / d
        else:
            inv[i] = 1e20 if d >= 0.0 else -1e20
    return inv


@ti.func
def ray_box_intersect(ray_origin, inv_ray_dir, box_min, box_max, t_max):
    """Slab test. Returns the entry distance, or T_MAX on a miss."""

    t1 = (box_min - ray_origin) * inv_ray_dir
    t2 = (box_max - ray_origin) * inv_ray_dir

    t_near = tm.min(t1, t2)
    t_far = tm.max(t1, t2)

    t_enter = tm.max(t_near[0], tm.max(t_near[1], t_near[2]))
    t_exit = tm.min(t_far[0], tm.min(t_far[1], t_far[2]))

    t_enter = tm.max(t_enter, 0.0)

    hit_t = T_MAX
    if t_exit > t_enter and t_enter <= t_max:
        hit_t = t_enter
    return hit_t


@ti.func
def ray_triangle_intersect(ray_origin, ray_dir, p0, p1, p2, t_max):
    """Moller-Trumbore. Returns vec3(t, u, v); t is T_MAX on a miss."""
    hit_t = T_MAX
    hit_u = 0.0
    hit_v = 0.0

    E1 = p1 - p0
    E2 = p2 - p0

    p = tm.cross(ray_dir, E2)
    det = tm.dot(p, E1)

    if abs(det) > 1e-12:
        inv_det = 1.0 / det
        T = ray_origin - p0

        u = tm.dot(p, T) * inv_det
        if 0.0 <= u <= 1.0:
            q = tm.cross(T, E1)
            v = tm.dot(q, ray_dir) * inv_det

            if v >= 0.0 and (u + v) <= 1.0:
                t = tm.dot(q, E2) * inv_det
                if RAY_EPS < t < t_max:
                    hit_t = t
                    hit_u = u
                    hit_v = v

    return tm.vec3(hit_t, hit_u, hit_v)
