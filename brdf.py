import taichi as ti 
import taichi.math as tm 
from texture import sample_albedo, sample_roughness, sample_specular

@ti.func 
def ggx_distribution(n, h, roughness):
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    cos_h = tm.max(0.0, tm.dot(n, h))
    denom = cos_h * cos_h * (alpha2 - 1.0) + 1.0
    return alpha2 / (tm.pi * denom * denom + 1e-8)


@ti.func
def smith_g1(n, v, roughness):
    alpha = roughness * roughness
    k = alpha / 2.0
    cos_v = tm.max(0.0, tm.dot(n, v))
    return cos_v / (cos_v * (1.0 - k) + k + 1e-8)


@ti.func
def smith_geometry(n, wi, wo, roughness):
    return smith_g1(n, wi, roughness) * smith_g1(n, wo, roughness)


@ti.func
def fresnel_schlick(cos_theta, f0):
    m = tm.clamp(1.0 - cos_theta, 0.0, 1.0)
    return f0 + (1.0 - f0) * (m * m * m * m * m)

@ti.func
def eval_specular_brdf(n, wi, wo, uv, roughness):
    f0 = sample_specular(uv) * 0.028  
    cos_i = tm.max(0.0, tm.dot(n, wi))
    cos_o = tm.max(0.0, tm.dot(n, wo))
    result = tm.vec3(0.0, 0.0, 0.0)
    if cos_i > 1e-4 and cos_o > 1e-4:
        h = tm.normalize(wi + wo)
        D = ggx_distribution(n, h, roughness)
        G = smith_geometry(n, wi, wo, roughness)
        Fr = fresnel_schlick(tm.max(0.0, tm.dot(wo, h)), f0)
        s = D * G * Fr / (4.0 * cos_i * cos_o + 1e-8)
        result = tm.vec3(s, s, s)
    return result



@ti.func
def brdf_pdf(n, wi, wo, roughness, prob_specular):
    cos_i = tm.max(0.0, tm.dot(n, wi))
    pdf_diffuse = cos_i / tm.pi

    pdf_specular = 0.0
    h = wi + wo
    if tm.dot(h, h) > 1e-8:
        h = tm.normalize(h)
        D = ggx_distribution(n, h, roughness)
        pdf_specular = D * tm.max(0.0, tm.dot(n, h)) / (4.0 * tm.max(1e-6, tm.dot(wo, h)))

    return prob_specular * pdf_specular + (1.0 - prob_specular) * pdf_diffuse