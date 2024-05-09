import mitsuba as mi 
import math
import drjit as dr
from utils import *

def shadow_masking_term_reflect (wi, wo, cos_theta_i, cos_theta_o, optical_depth, s_mat) : 
    """
    When gamma_wi + gamma_wo is positive, which is expected in the reflecting case, 
    max value of this term can be optical_depth. 

    If negative, it is unbounded. This matches the data that I printed out. Only when cos_theta_i is negative,
    it changes

    This is activated when cos_theta_i * cos_theta_o > 0
    """
    sigma_wi = sggx_projected_area(wi, s_mat)
    sigma_wo = sggx_projected_area(wo, s_mat)

    cos_theta_o = dr.mulsign(cos_theta_o, cos_theta_i)
    cos_theta_i = dr.mulsign(cos_theta_i, cos_theta_i)

    gamma_wi = sigma_wi / cos_theta_i
    gamma_wo = sigma_wo / cos_theta_o

    return (1.0 - dr.exp(-optical_depth * (gamma_wi + gamma_wo))) / (gamma_wi + gamma_wo)

def shadow_masking_term_transmit (wi, wo, cos_theta_i, cos_theta_o, optical_depth, s_mat): 
    # TODO: Check offline whether these terms are sensible
    sigma_wi = sggx_projected_area(wi, s_mat)
    sigma_wo = sggx_projected_area(wo, s_mat)

    cos_theta_o = dr.mulsign(cos_theta_o, cos_theta_i)
    cos_theta_i = dr.mulsign(cos_theta_i, cos_theta_i)

    gamma_wi = sigma_wi / cos_theta_i
    gamma_wo = sigma_wo / cos_theta_o

    extra_term = dr.exp(optical_depth * gamma_wo)
    original_term = (1.0 - dr.exp(-optical_depth * (gamma_wi + gamma_wo))) / (gamma_wi + gamma_wo)

    return original_term * extra_term

class SimpleSpongeCake (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__ (props)  
        self.base_color = props['base_color'] 
        self.optical_depth = mi.Float(props['optical_depth']) # the product T\rho
        self.alpha = mi.Float(props['alpha'])
        self.surface_or_fiber = mi.Bool(props['surface_or_fiber'])

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample (self, ctx, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 

        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) # surface type matrix. Later we'll rotate it and all that.
        S_fibr = dr.diag(mi.Vector3f(1., 1., alpha * alpha)) # surface type matrix. Later we'll rotate it and all that.

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)

        sh_frame = mi.Frame3f(si.wi)
        # h = sggx_sample(si.sh_frame, sample2, S) # is this generated sample in the local frame or global frame? 
        # TODO: looking at the docs at https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#sggx_sample, it seems that
        # the sh_frame is constructed as mi.Frame3f(si.wi). I checked via PDB that this is different from si.sh_frame
        # even if this is correct, the question remains whether the h is correct. I think so. It'll be in the same coordinate system as wi now.
        # which is the local coordinate frame. Even if we look at the visualizations now, they see better.
        h = sggx_sample(sh_frame, sample2, S)
        D = sggx_pdf(h, S)
        
        wo = mi.reflect(si.wi, h) 
        F = mi.Color3f(self.base_color) + (1.0 - mi.Color3f(self.base_color)) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        selected_r = (cos_theta_i * cos_theta_o > 0.) # reflecting when the two have the same sign

        G_r = shadow_masking_term_reflect(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)
        G_t = shadow_masking_term_transmit(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)

        G = dr.select(selected_r, G_r, G_t)

        bs.wo = wo
        bs.eta = 1.
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
        bs.pdf = D / (4. * dr.abs(dr.dot(bs.wo, h))) # multiply by Jacobian

        active = active & dr.neq(cos_theta_i, 0.0) & dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) 

        f_sponge_cake = (F * D * G) / (4. * dr.abs(cos_theta_i))

        R = dr.abs(dr.dot(bs.wo, h)) / dr.abs(cos_theta_i)
        
        weight = (f_sponge_cake / bs.pdf) 
        active = active & (dr.all(dr.isfinite(weight)))
        weight = weight & active 

        # sigma_wi = sggx_projected_area(si.wi, S)
        # sigma_wo = sggx_projected_area(wo, S)

        # cos_theta_o = dr.mulsign(cos_theta_o, cos_theta_i)
        # cos_theta_i = dr.mulsign(cos_theta_i, cos_theta_i)

        # gamma_wi = sigma_wi / cos_theta_i
        # gamma_wo = sigma_wo / cos_theta_o

        # extra_term = dr.exp(self.optical_depth * gamma_wo)
        # original_term = (1.0 - dr.exp(-self.optical_depth * (gamma_wi + gamma_wo))) / (gamma_wi + gamma_wo)
        # print(f'{self.optical_depth},{weight[0]},{weight[1]},{weight[2]},{G_r},{G_t},{F[0]},{F[1]},{F[2]},{R},{original_term},{extra_term},{gamma_wi},{gamma_wo},{selected_r}')
            
        return (bs, weight)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        callback.put_parameter('base_color', self.base_color, mi.ParamFlags.Differentiable)
        callback.put_parameter('optical_depth', self.optical_depth, mi.ParamFlags.Differentiable)
        callback.put_parameter('alpha', self.alpha, mi.ParamFlags.Differentiable)
        callback.put_parameter('surface_or_fiber', self.surface_or_fiber, mi.ParamFlags.Discontinuous | mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return ('SimpleSpongeCake[]')
