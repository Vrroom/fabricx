import mitsuba as mi 
import math
import drjit as dr
from utils import *

def shadow_masking_term_reflect (wi, wo, cos_theta_i, cos_theta_o, optical_depth, s_mat) : 
    # TODO: Check offline whether these terms are sensible
    sigma_wi = sggx_projected_area(wi, s_mat)
    sigma_wo = sggx_projected_area(wo, s_mat)

    gamma_wi = sigma_wi / cos_theta_i
    gamma_wo = sigma_wo / cos_theta_o

    return (1.0 - dr.exp(-optical_depth * (gamma_wi + gamma_wo))) / (gamma_wi + gamma_wo)

def shadow_masking_term_transmit (wi, wo, cos_theta_i, cos_theta_o, optical_depth, s_mat): 
    # TODO: Check offline whether these terms are sensible
    sigma_wi = sggx_projected_area(wi, s_mat)
    sigma_wo = sggx_projected_area(wo, s_mat)

    gamma_wi = sigma_wi / cos_theta_i
    gamma_wo = sigma_wo / cos_theta_o

    extra_term = dr.exp(optical_depth * gamma_wo)
    original_term = (1.0 - dr.exp(-optical_depth * (gamma_wi + gamma_wo))) / (gamma_wi + gamma_wo)

    return original_term * extra_term

class SimpleSpongeCake (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__ (props)  
        self.base_color = props['base_color'] 
        self.optical_depth = props['optical_depth'] # the product T\rho
        self.roughness = props['roughness'] 

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample (self, ctx, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        alpha = dr.maximum(0.0001, self.roughness**2)

        S = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1)) # surface type matrix. Later we'll rotate it and all that.

        h = sggx_sample(si.sh_frame, sample2, S)
        D = sggx_pdf(h, S)
        
        wo = mi.reflect(si.wi, h) 
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        F = mi.Color3f(self.base_color) + (1.0 - mi.Color3f(self.base_color)) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

        selected_r = (cos_theta_o > 0.)

        G_r = shadow_masking_term_reflect(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)
        G_t = shadow_masking_term_transmit(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)

        G = dr.select(selected_r, G_r, G_t)

        bs.wo = wo
        bs.eta = 1.
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
        bs.pdf = D / 4. * dr.abs(dr.dot(bs.wo, h)) # multiply by Jacobian

        active = active & (cos_theta_i > 0.) & dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) # only allow outward for now.

        f_sponge_cake = F * D * G / (4 * dr.abs(cos_theta_i))

        weight = dr.clamp(f_sponge_cake / bs.pdf, 0, 3)
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
        return ('SimpleSpongeCake[]')
