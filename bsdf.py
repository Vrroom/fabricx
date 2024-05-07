import mitsuba as mi 
import math
import drjit as dr
from utils import *

def diffuse_lobe_sample (wi, sample1, sample2, *args, **kwargs) : 
    wo  = mi.warp.square_to_cosine_hemisphere(sample2)
    pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
    return wo, pdf, True, mi.UInt32(+mi.BSDFFlags.DiffuseReflection), dict()

def metal_lobe_sample (wi, sample1, sample2, anisotropic, roughness, *args, **kwargs) : 
    aspect = dr.sqrt(1 - 0.9 * anisotropic)
    alpha_x = dr.maximum(0.0001, roughness ** 2 / aspect)
    alpha_y = dr.maximum(0.0001, roughness ** 2 * aspect)

    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

    m, Dm = distr.sample(wi, sample2)
    wo = mi.reflect(wi, m)
    pdf = Dm / (4. * dr.dot(wo, m)) # change of variable jacobian

    Gm = distr.G(wi, wo, m)

    active = dr.neq(pdf, 0.) & (mi.Frame3f.cos_theta(wo) > 0.)
    return wo, pdf, active, mi.UInt32(+mi.BSDFFlags.GlossyReflection), dict()

def glass_lobe_sample (wi, sample1, sample2, anisotropic, roughness, eta, cos_theta_i, *args, **kwargs) : 
    aspect = dr.sqrt(1 - 0.9 * anisotropic)
    alpha_x = dr.maximum(0.0001, roughness ** 2 / aspect)
    alpha_y = dr.maximum(0.0001, roughness ** 2 * aspect)

    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

    m, Dg = distr.sample(dr.mulsign(wi, cos_theta_i), sample2)
    active = dr.neq(Dg, 0.)

    Fg, cos_theta_t, eta_it, eta_ti = mi.fresnel(dr.dot(wi, m), eta)

    selected_r = (sample1 <= Fg) & active

    pdf = Dg * dr.select(selected_r, Fg, 1. - Fg)

    wo = dr.select(selected_r, mi.reflect(wi, m), mi.refract(wi, m, cos_theta_t, eta_ti))
    dwh_dwo = dr.select(selected_r, 
        1. / (4. * dr.dot(wo, m)),
        (dr.sqr(eta_it) * dr.dot(wo, m)) / dr.sqr(dr.dot(wi, m) + eta_it * dr.dot(wo, m))
    )
    pdf *= dr.abs(dwh_dwo) # change of variable jacobian accounting for both possibilities

    sampled_type = dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
    Gg = distr.G(wi, wo, m)

    return wo, pdf, active, sampled_type, dict(Dg=Dg, Gg=Gg, Fg=Fg, selected_r=selected_r, eta_it=eta_it, h=m, dwh_dwo=dwh_dwo)

def clearcoat_lobe_sample(wi, sample1, sample2, clearcoat_gloss) : 
    alpha_g = (1. - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001
    alpha_g_sq = alpha_g ** 2

    cos_h_el = dr.sqrt((1. - dr.power(alpha_g_sq, 1 - sample2[0])) / (1 - alpha_g_sq))
    sin_h_el = dr.sqrt(1.0 - cos_h_el ** 2)

    h_az = 2 * math.pi * sample2[1]

    hx = sin_h_el * dr.cos(h_az)
    hy = sin_h_el * dr.sin(h_az)
    hz = cos_h_el

    h = mi.Vector3f(hx, hy, hz) 

    wo = mi.reflect(wi, h) 
    
    Dc = (alpha_g_sq - 1.) / (dr.pi * dr.log(alpha_g_sq) * (1. + (alpha_g_sq - 1) * (hz ** 2))) 

    pdf = Dc / (4. * dr.dot(wo, h))
    sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

    active = dr.neq(pdf, 0.) 

    return wo, pdf, active, sampled_type, dict() 

def diffuse_bsdf (wi, wo, cos_theta_i, cos_theta_o, roughness, base_color, subsurface) : 
    """ evaluate bsdf * forshortening factor (|n.wo|) """ 
    h = half_vec(wi, wo)

    Fd90    = 0.5 + 2. * roughness * (dr.dot(h, wo) ** 2)

    Fd_wi   = (1 + (Fd90 - 1) * ((1 - dr.abs(cos_theta_i)) ** 5))
    Fd_wo   = (1 + (Fd90 - 1) * ((1 - dr.abs(cos_theta_o)) ** 5))

    f_base_diffuse = (mi.Color3f(base_color) / dr.pi) * Fd_wi * Fd_wo * dr.abs(cos_theta_o)

    FSS90 = roughness * (dr.dot(h, wo) ** 2)
    FSS_wi = (1. + (FSS90 - 1.) * (1 - dr.abs(cos_theta_i)) ** 5)
    FSS_wo = (1. + (FSS90 - 1.) * (1 - dr.abs(cos_theta_o)) ** 5)

    rcp = (1. / (dr.abs(cos_theta_i) + dr.abs(cos_theta_o))) - 0.5
    f_subsurface = ((1.25 * mi.Color3f(base_color)) / dr.pi) \
            * (FSS_wi * FSS_wo * rcp + 0.5) \
            * dr.abs(cos_theta_o)

    f_diffuse = ((1 - subsurface) * f_base_diffuse + subsurface * f_subsurface) 
    return f_diffuse

def metal_bsdf(wi, wo, cos_theta_i, cos_theta_o, base_color, anisotropic, roughness) :
    aspect = dr.sqrt(1 - 0.9 * anisotropic)
    alpha_x = dr.maximum(0.0001, roughness ** 2 / aspect)
    alpha_y = dr.maximum(0.0001, roughness ** 2 * aspect)

    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)
    h = half_vec(wi, wo)
    Dm = distr.pdf(wi, h)
    Gm = distr.G(wi, wo, h)
    Fm = mi.Color3f(base_color) + (1.0 - mi.Color3f(base_color)) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)
    return (Fm * Dm * Gm) / (4. * dr.abs(cos_theta_i))

def modified_metal_bsdf (wi, wo, cos_theta_i, cos_theta_o, base_color, anisotropic, roughness, specular_tint, specular, metallic) :
    aspect = dr.sqrt(1 - 0.9 * anisotropic)
    alpha_x = dr.maximum(0.0001, roughness ** 2 / aspect)
    alpha_y = dr.maximum(0.0001, roughness ** 2 * aspect)

    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

    h = half_vec(wi, wo)

    Dm = distr.pdf(wi, h)
    Gm = distr.G(wi, wo, h)

    base_color = mi.Color3f(base_color)
    white_color = mi.Color3f([1., 1., 1.])

    lum = mi.luminance(base_color)

    C_tint = dr.select(lum > 0, base_color / lum, white_color)
    Ks = (1 - specular_tint) + specular_tint * C_tint
    R0 = (1.5 - 1) ** 2 / (1.5 + 1) ** 2
    C0 = specular * R0 * (1 - metallic) * Ks + metallic * base_color

    Fm = C0 + (1 - C0) * (1 - dr.abs(dr.dot(h, wo))) ** 5 

    f_metal = Fm * Dm * Gm / (4. * dr.abs(cos_theta_i))

    return f_metal

def glass_bsdf (wi, wo, cos_theta_i, cos_theta_o, base_color, anisotropic, roughness, eta, selected_r, h) : 
    aspect = dr.sqrt(1 - 0.9 * anisotropic)
    alpha_x = dr.maximum(0.0001, roughness ** 2 / aspect)
    alpha_y = dr.maximum(0.0001, roughness ** 2 * aspect)

    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)
    Dg = distr.pdf(dr.mulsign(wi, cos_theta_i), h)
    Gg = distr.G(wi, wo, h)
    Fg, _, eta_it, eta_ti = mi.fresnel(dr.dot(wi, h), eta)

    f_glass_r = (mi.Color3f(base_color) * Fg * Dg * Gg) / (4 * dr.abs(cos_theta_i))
    f_glass_t = (dr.sqrt(mi.Color3f(base_color)) * (1 - Fg) * Dg * Gg * dr.abs(dr.dot(h, wo) * dr.dot(h, wi))) \
            / (dr.abs(cos_theta_i) * (dr.dot(h, wi) + eta_it * dr.dot(h, wo)) ** 2)
    return dr.select(selected_r, f_glass_r, f_glass_t)

def clearcoat_bsdf (wi, wo, cos_theta_i, cos_theta_o, clearcoat_gloss) : 
    alpha_g = (1. - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001
    alpha_g_sq = alpha_g ** 2
    h = half_vec(wi, wo)
    R0 = (1.5 - 1) ** 2 / (1.5 + 1) ** 2
    Fc = R0 + (1 - R0) * (1. - dr.abs(dr.dot(h, wo))) ** 5
    Dc = (alpha_g_sq - 1.) / (dr.pi * dr.log(alpha_g_sq) * (1. + (alpha_g_sq - 1) * (h.z ** 2))) 
    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, 0.25, 0.25)
    Gc = distr.G(wi, wo, h)
    f_clearcoat = Fc * Dc * Gc / (4 * dr.abs(cos_theta_i)) 
    return f_clearcoat

def sheen_bsdf (wi, wo, cos_theta_i, cos_theta_o, base_color, sheen_tint) : 
    base_color = mi.Color3f(base_color)
    white_color = mi.Color3f([1., 1., 1.])

    lum = mi.luminance(base_color)

    C_tint = dr.select(lum > 0, base_color / lum, white_color)
    C_sheen = (1 - sheen_tint) + sheen_tint * C_tint

    h = half_vec(wi, wo) 
    f_sheen = C_sheen * ((1 - dr.abs(dr.dot(h, wo))) ** 5) * dr.abs(wo)
    return f_sheen


class DisneyDiffuseBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.base_color = props['base_color']
        self.roughness = props['roughness']
        self.subsurface = props['subsurface'] 

        reflection_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags] 
        self.m_flags = reflection_flags 

    def sample (self, ctx, si, sample1, sample2, active) : 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        wo, pdf, act, sampled_type, misc = diffuse_lobe_sample(si.wi, sample1, sample2)
        active = active & act
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        f_diffuse = diffuse_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.roughness, self.base_color, self.subsurface)

        weight = f_diffuse / pdf

        active = active & (cos_theta_i > 0.) & (cos_theta_o > 0.)

        bs = mi.BSDFSample3f() 
        bs.pdf = pdf
        bs.sampled_component = mi.UInt32(0) 
        bs.sampled_type = sampled_type
        bs.wo = wo
        bs.eta = 1.0 

        return (bs, weight & active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        callback.put_parameter('base_color', self.base_color, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return ('DisneyDiffuseBSDF[]')

class DisneyMetalBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.base_color = props['base_color']
        self.roughness = props['roughness']
        self.anisotropic = props['anisotropic']

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags] 
        self.m_flags = reflection_flags 

    def sample (self, ctx, si, sample1, sample2, active) : 
        # based on the rough conductor bsdf in mitsuba.cpp
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        wo, pdf, act, sampled_type, misc = metal_lobe_sample(si.wi, sample1, sample2, self.anisotropic, self.roughness)
        active = active & act
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        f_metal = metal_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.anisotropic, self.roughness)

        weight = f_metal / pdf

        bs = mi.BSDFSample3f() 
        bs.pdf = pdf
        bs.sampled_component = mi.UInt32(0) 
        bs.sampled_type = sampled_type
        bs.wo = wo
        bs.eta = 1.0 

        active = active & (cos_theta_i > 0.) & (cos_theta_o > 0.)

        return (bs, (weight) & active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        callback.put_parameter('base_color', self.base_color, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        Return ('DisneyMetalBSDF[]')

class DisneyGlassBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.base_color = props['base_color']
        self.roughness = props['roughness']
        self.anisotropic = props['anisotropic']
        self.eta = props['eta']

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags, transmission_flags] 
        self.m_flags = reflection_flags | transmission_flags

    def sample (self, ctx, si, sample1, sample2, active) : 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active = active & dr.neq(cos_theta_i, 0.) # now since we are transmitting, a ray can hit from inside
 
        wo, pdf, act, sampled_type, misc = glass_lobe_sample(si.wi, sample1, sample2, self.anisotropic, self.roughness, self.eta, cos_theta_i)
        active = active & act
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        f_glass = glass_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.anisotropic, self.roughness, self.eta, misc['selected_r'], misc['h'])
 
        weight = f_glass / pdf
 
        selected_r = misc['selected_r']
 
        bs = mi.BSDFSample3f() 
        bs.pdf = pdf
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = sampled_type
        bs.wo = wo
        bs.eta = dr.select(selected_r, 1.0, misc['eta_it'])
 
        return (bs, (weight) & active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        callback.put_parameter('base_color', self.base_color, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        Return ('DisneyGlassBSDF[]')

class DisneyClearcoatBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.clearcoat_gloss = props['clearcoat_gloss']

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags] 
        self.m_flags = reflection_flags 

    def sample (self, ctx, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        active = active & (cos_theta_i > 0.)

        wo, pdf, active, sampled_type, misc = clearcoat_lobe_sample(si.wi, sample1, sample2, self.clearcoat_gloss)

        bs.wo = wo
        bs.eta = 1.
        bs.pdf = pdf
        bs.sampled_component = 0
        bs.sampled_type = sampled_type

        active = active & dr.neq(bs.pdf, 0.) & (mi.Frame3f.cos_theta(bs.wo) > 0. )

        cos_theta_o = mi.Frame3f.cos_theta(wo) 
        f_clearcoat = clearcoat_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.clearcoat_gloss)
        weight = f_clearcoat / pdf

        return (bs, (weight) & active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        Return ('DisneyClearcoatBSDF[]')

class DisneySheenBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.sheen_tint = props['sheen_tint']
        self.base_color = props['base_color'] 

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags] 
        self.m_flags = reflection_flags 

    def sample (self, ctx, si, sample1, sample2, active) : 
        wo, pdf, act, sampled_type, misc = diffuse_lobe_sample (si.wi, sample1, sample2)
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        cos_theta_o = mi.Frame3f.cos_theta(wo) 

        active = active & (cos_theta_i > 0.) & act & (cos_theta_o > 0.) 

        f_sheen = sheen_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.sheen_tint)

        bs = mi.BSDFSample3f() 
        bs.wo = wo
        bs.eta = 1.
        bs.pdf = pdf
        bs.sampled_component = 0
        bs.sampled_type = sampled_type

        weight = f_sheen / bs.pdf

        return (bs, (weight) & active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        Return ('DisneySheenBSDF[]')

class DisneyPrincipledBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__ (props) 

        self.base_color = props['base_color'] 
        self.specular_transmission = props['specular_transmission']
        self.metallic = props['metallic'] 
        self.subsurface = props['subsurface']
        self.specular = props['specular']
        self.roughness = props['roughness']
        self.specular_tint = props['specular_tint']
        self.anisotropic = props['anisotropic']
        self.sheen = props['sheen']
        self.sheen_tint = props['sheen_tint']
        self.clearcoat = props['clearcoat']
        self.clearcoat_gloss = props['clearcoat_gloss']
        self.eta = props['eta']

        self.diffuseWeight = (1 - self.metallic) * (1 - self.specular_transmission)
        self.sheenWeight = (1 - self.metallic) * self.sheen 
        self.metalWeight = (1 - self.specular_transmission * (1 - self.metallic))
        self.glassWeight = (1 - self.metallic) * self.specular_transmission
        self.clearcoatWeight = 0.25 * self.clearcoat

        total = sum([self.glassWeight, self.diffuseWeight, self.metalWeight, self.clearcoatWeight])

        self.glassProb = self.glassWeight / total
        self.diffuseProb = self.diffuseWeight / total
        self.metalProb = self.metalWeight / total
        self.clearcoatProb = self.clearcoatWeight / total

        diffuse_reflection_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        glossy_reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        glossy_transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [diffuse_reflection_flags, glossy_reflection_flags, glossy_transmission_flags] 
        self.m_flags = diffuse_reflection_flags | glossy_reflection_flags | glossy_transmission_flags

    def sample (self, ctx, si, sample1, sample2, active) : 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        inside = cos_theta_i < 0.

        if inside : 
            # only glass is active inside the surface
            active = active & dr.neq(cos_theta_i, 0.) # now since we are transmitting, a ray can hit from inside
     
            wo, pdf, act, sampled_type, misc = glass_lobe_sample(si.wi, sample1, sample2, self.anisotropic, self.roughness, self.eta, cos_theta_i)
            active = active & act
            cos_theta_o = mi.Frame3f.cos_theta(wo)
            f_glass = glass_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.anisotropic, self.roughness, self.eta, misc['selected_r'], misc['h'])
     
            weight = f_glass / pdf
     
            selected_r = misc['selected_r']
     
            bs = mi.BSDFSample3f() 
            bs.pdf = pdf
            bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
            bs.sampled_type = sampled_type
            bs.wo = wo
            bs.eta = dr.select(selected_r, 1.0, misc['eta_it'])
     
            return (bs, (weight) & active)
        else :
            reflecting = True
            if sample1 < self.glassProb : 
                sample_new = sample1 / self.glassProb
                wo, pdf, act, sampled_type, misc = glass_lobe_sample(si.wi, sample_new, sample2, self.anisotropic, self.roughness, self.eta, cos_theta_i)
                reflecting = reflecting & misc['selected_r']
                pdf *= self.glassProb
            elif sample1 < self.glassProb + self.diffuseProb :
                wo, pdf, act, sampled_type, misc = diffuse_lobe_sample(si.wi, sample1, sample2)
                pdf *= self.diffuseProb
            elif sample1 < self.glassProb + self.diffuseProb + self.metalProb : 
                wo, pdf, act, sampled_type, misc = metal_lobe_sample(si.wi, sample1, sample2, self.anisotropic, self.roughness) 
                pdf *= self.metalProb
            else :
                wo, pdf, act, sampled_type, misc = clearcoat_lobe_sample(si.wi, sample1, sample2, self.clearcoat_gloss)
                pdf *= self.clearcoatProb

            active = active & act
            cos_theta_o = mi.Frame3f.cos_theta(wo)
            
            if reflecting : 
                f_diffuse = diffuse_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.roughness, self.base_color, self.subsurface)
                f_metal = modified_metal_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, 
                        self.anisotropic, self.roughness, self.specular_tint, self.specular, self.metallic)
                f_glass = glass_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.anisotropic, 
                        self.roughness, self.eta, misc['selected_r'], misc['h'])
                f_sheen = sheen_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.sheen_tint)
                f_clearcoat = clearcoat_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.clearcoat_gloss)
                
                f_principled = (self.diffuseWeight * f_diffuse) \
                        + (self.sheenWeight * f_sheen) \
                        + (self.metalWeight * f_metal) \
                        + (self.glassWeight * f_glass) \
                        + (self.clearcoatWeight * f_clearcoat)

                weight = f_principled / pdf

                active = active & (cos_theta_i > 0.) & (cos_theta_o > 0.)

                bs = mi.BSDFSample3f() 
                bs.pdf = pdf
                bs.sampled_component = mi.UInt32(0) 
                bs.sampled_type = sampled_type
                bs.wo = wo
                bs.eta = 1.0 

                return (bs, weight & active)

            else : 
                f_glass = glass_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.base_color, self.anisotropic, self.roughness, self.eta, misc['selected_r'], misc['h'])
         
                weight = f_glass / pdf
         
                selected_r = misc['selected_r']
         
                bs = mi.BSDFSample3f() 
                bs.pdf = pdf
                bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
                bs.sampled_type = sampled_type
                bs.wo = wo
                bs.eta = dr.select(selected_r, 1.0, misc['eta_it'])
         
                return (bs, (weight) & active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return ('DisneyPrincipledBSDF[]')
