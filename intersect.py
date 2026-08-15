import taichi as ti 
import taichi.math as tm 

@ti.func
def ray_box_intersect(ray_origin, inv_ray_dir, box_min, box_max):
    t1 = (box_min - ray_origin) * inv_ray_dir
    t2 = (box_max - ray_origin) * inv_ray_dir

    t_near = tm.min(t1, t2)
    t_far = tm.max(t1, t2)

    t_enter = tm.max(t_near[0], tm.max(t_near[1], t_near[2]))
    t_exit = tm.min(t_far[0], tm.min(t_far[1], t_far[2]))

    hit_point = 1e30
    if t_exit >= tm.max(t_enter, 0.0):
        hit_point = tm.max(t_enter, 0.0)
    return hit_point


@ti.func
def ray_triangle_intersection(ray_origin, ray_dir, p0, p1, p2):
    EPS = 1e-8
    hit_t = 1e30
    hit_u = 0.0
    hit_v = 0.0

    E1 = p1 - p0
    E2 = p2 - p0 

    p = tm.cross(ray_dir, E2)
    det = tm.dot(p, E1)

    if abs(det) > EPS:
        inv_det = 1/det 
        T = ray_origin - p0
        u_top =  tm.dot(p, T)
        u = u_top * inv_det

        if u >= 0.0 and u <= 1.0:
            q = tm.cross(T, E1)
            v_top = tm.dot(q, ray_dir)
            v = v_top * inv_det

            if v >= 0.0 and (u + v) <= 1.0:
                t_top = tm.dot(q, E2)
                t = t_top * inv_det

                if t > EPS :
                    hit_t = t
                    hit_u = u 
                    hit_v = v 
                
    return tm.vec3(hit_t, hit_u, hit_v)


   


    