import mitsuba as mi 
from PIL import Image
import numpy as np
import math
import drjit as dr
from utils import *
from asg import (
    benny_rodrigues_rot_formula_dr, 
    spherical_to_euclidean_dr,
    euclidean_to_spherical_dr,
    asg_dr
)

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

    return (1.0 - dr.exp(-optical_depth * (gamma_wi + gamma_wo + 1e-8))) / (gamma_wi + gamma_wo + 1e-8)

def shadow_masking_term_transmit (wi, wo, cos_theta_i, cos_theta_o, optical_depth, s_mat): 
    # TODO: Check offline whether these terms are sensible
    sigma_wi = sggx_projected_area(wi, s_mat)
    sigma_wo = sggx_projected_area(wo, s_mat)

    cos_theta_o = dr.mulsign(cos_theta_o, cos_theta_i)
    cos_theta_i = dr.mulsign(cos_theta_i, cos_theta_i)

    gamma_wi = sigma_wi / cos_theta_i
    gamma_wo = sigma_wo / cos_theta_o

    extra_term = dr.exp(optical_depth * gamma_wo)
    original_term = (1.0 - dr.exp(-optical_depth * (gamma_wi + gamma_wo + 1e-8))) / (gamma_wi + gamma_wo + 1e-8)

    return original_term * extra_term

def sample_specular (sample2, wi, S) : 
    sh_frame = mi.Frame3f(wi) 
    h = sggx_sample(sh_frame, sample2, S)
    D = sggx_pdf(h, S)
    wo = mi.reflect(wi, h)
    pdf = D / (4. * sggx_projected_area(wi, S))
    return h, D, wo, pdf

def sample_diffuse (pcg, sample2, wi, S) : 
    sh_frame = mi.Frame3f(wi) 
    h = sggx_sample(sh_frame, sample2, S) # can't use the same sample2. Has to be independent.
    D = sggx_pdf(h, S)
    sample_iid = mi.Point2f(pcg.next_float32(), pcg.next_float32())
    wo_local = mi.warp.square_to_cosine_hemisphere(sample_iid) # I verified via the equation in SGGX paper (26) that this is indeed correct
    wo = mi.Frame3f(h).to_world(wo_local)
    pdf = dr.dot(wo, h) / dr.pi # need to worry about the pdf here
    return h, D, wo, pdf

def sample_diffuse_jin (sample2, wi) : 
    wo = mi.warp.square_to_cosine_hemisphere(sample2) 
    pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
    return mi.Vector3f(0, 0, 1), 1.0, wo, pdf

class SimpleSpongeCake (mi.BSDF) : 

    def __init__ (self, props, *args, **kwargs) : 
        super().__init__ (props)  
        self.base_color = props['base_color'] 
        self.optical_depth = mi.Float(props['optical_depth']) # the product T\rho
        self.alpha = mi.Float(props['alpha'])
        self.surface_or_fiber = mi.Bool(props['surface_or_fiber'])

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

        # HACK: force reference so that mitsuba doesn't complain and we can override from cmd line
        texture_file = props['texture']
        normal_map_file = props['normal_map']
        tangent_map_file = props['tangent_map']

        self.pcg = mi.PCG32()

    def sample (self, ctx, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 

        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) # surface type matrix. Later we'll rotate it and all that.
        S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.)) # fiber type matrix. Later we'll rotate it and all that.
        # S_fibr = dr.diag(mi.Vector3f(1., 1., alpha * alpha)) # fiber type matrix. Later we'll rotate it and all that.

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)

        specular_or_diffuse = sample1 < 1.0

        h, D, wo, pdf = sample_specular(sample2, si.wi, S)
        h_, D_, wo_, pdf_ = sample_diffuse(self.pcg, sample2, si.wi, S)

        h = dr.select(specular_or_diffuse, h, h_)
        D = dr.select(specular_or_diffuse, D, D_)
        wo = dr.select(specular_or_diffuse, wo, wo_)
        pdf = dr.select(specular_or_diffuse, pdf, pdf_)


        color = mi.Color3f(self.base_color) 
        F = color + (1.0 - color) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

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
        bs.pdf = pdf 

        active = active & dr.neq(cos_theta_i, 0.0) & dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) 


        f_sponge_cake = (F * D * G) / (4. * dr.abs(cos_theta_i))

        weight = (f_sponge_cake / bs.pdf) 
        active = active & (dr.all(dr.isfinite(weight)))
        weight = weight & active 
            
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

class SpongeCake (mi.BSDF) : 

    def __init__ (self, props, *args, texture=None, normal_map=None, 
            tangent_map=None, perturb_specular=False, delta_transmission=False) : 
        super().__init__ (props)  
        self.base_color = props['base_color'] 
        self.optical_depth = mi.Float(props['optical_depth']) # the product T\rho
        self.alpha = mi.Float(props['alpha'])
        self.surface_or_fiber = mi.Bool(props['surface_or_fiber'])
        self.w = mi.Float(props['w'])

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        delta_transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [reflection_flags, transmission_flags, delta_transmission_flags]
        self.m_flags = reflection_flags | transmission_flags | delta_transmission_flags

        self.pcg = mi.PCG32()

        self.perturb_specular = mi.Bool(perturb_specular)
        self.delta_transmission = mi.Bool(delta_transmission)

        # HACK: force reference so that mitsuba doesn't complain and we can override from cmd line
        texture_file = props['texture']
        normal_map_file = props['normal_map']
        tangent_map_file = props['tangent_map']

        normal_map_file = normal_map if normal_map is not None else props['normal_map']
        tangent_map_file = tangent_map if tangent_map is not None else props['tangent_map']
        texture_file = texture if texture is not None else props['texture']

        self.bent_normal_map = mi.Texture2f(mi.TensorXf(fix_map(np.array(Image.open('misc/bent_normal_map.png').convert('RGB')))))
        self.asg_params = mi.Texture2f(mi.TensorXf(np.load('misc/asg_params.npy')))

        nm = np.array(Image.open(normal_map_file).convert('RGB'))
        tm = np.array(Image.open(tangent_map_file).convert('RGB'))

        nm, tm = fix_normal_and_tangent_map(nm, tm)

        self.normal_map = mi.Texture2f(mi.TensorXf(nm))
        self.tangent_map = mi.Texture2f(mi.TensorXf(tm))

        texture_map = np.array(Image.open(texture_file)) / 255.
        self.texture = mi.Texture2f(mi.TensorXf(texture_map[..., :3]))
        delta_map = np.ones(tuple(texture_map.shape[:-1]) + (1,)) if not delta_transmission else texture_map[..., 3:]
        assert delta_map.shape[-1] == 1, "Texture has no or wrong delta transmission channel."
        # 0 means transmit, 1 means go through
        self.delta_transmission_map = mi.Texture2f(mi.TensorXf(delta_map))

    def sample (self, ctx, si, sample1, sample2, active) : 
        """ 
        TODOs:
        3. Perturb the maps
        4. Don't do closed form normal and tangent map. More flexible to compute it using geometry.
        7. Learn about LEAN and LEADR Maps
        9. Jin et al also perturb the specular weight by some random variable. It might be interesting to use Musgrave texture here. Because it does look like terrain
        10. Check differentiability
        11. Add documentation/tutorial
        13. Verify that I'm correctly combining lobes
        """
        bs = mi.BSDFSample3f() 
        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) 
        S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.)) 

        bent_normal = mi.Vector3f(self.bent_normal_map.eval(si.uv))
        asg_params = mi.Vector3f(self.asg_params.eval(si.uv))

        normal = mi.Vector3f(self.normal_map.eval(si.uv))
        tangent = mi.Vector3f(self.tangent_map.eval(si.uv))

        delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(si.uv))
        selected_dt = sample1 > delta_transmission.x
        
        normal = normal / dr.norm(normal)
        tangent= tangent / dr.norm(tangent)

        tangent = tangent - dr.sum(tangent * normal) * normal

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)

        S = rotate_s_mat(S, normal, tangent)

        specular_or_diffuse = sample1 < 1.0

        h, D, wo, pdf = sample_specular(sample2, si.wi, S)
        h_, D_, wo_, pdf_ = sample_diffuse(self.pcg, sample2, si.wi, S)
        h_dt, D_dt, wo_dt, pdf_dt = mi.Vector3f(0,0,1), 1.0, -si.wi, 1.0

        h = dr.select(specular_or_diffuse, h, h_)
        D = dr.select(specular_or_diffuse, D, D_)
        wo = dr.select(specular_or_diffuse, wo, wo_)
        pdf = dr.select(specular_or_diffuse, pdf, pdf_)

        # delta transmission
        h = dr.select(selected_dt, h_dt, h)
        D = dr.select(selected_dt, D_dt, D)
        wo = dr.select(selected_dt, wo_dt, wo)
        pdf = dr.select(selected_dt, pdf_dt, pdf)

        color = mi.Color3f(self.texture.eval(si.uv)) 
        F = color + (1.0 - color) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        selected_r = (cos_theta_i * cos_theta_o > 0.) # reflecting when the two have the same sign

        G_r = shadow_masking_term_reflect(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)
        G_t = shadow_masking_term_transmit(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)

        G = dr.select(selected_r, G_r, G_t)

        bs.wo = wo
        bs.eta = 1.
        bs.sampled_component = dr.select(selected_dt, 
            mi.UInt32(2), 
            dr.select(selected_r, mi.UInt32(0), mi.UInt32(1)))
        bs.sampled_type = dr.select(selected_dt, 
            mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
            dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.GlossyTransmission)))
        bs.pdf = pdf 

        # EXPERIMENT WITH VISIBILITY
        di = euclidean_to_spherical_dr(si.wi) 
        do = euclidean_to_spherical_dr(wo) 
        bent_normal = bent_normal / dr.norm(bent_normal)
        mu = euclidean_to_spherical_dr(bent_normal)
        V_i = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, di)
        V_o = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, do)

        v_threshold = 0.5
        active = active & dr.neq(cos_theta_i, 0.0) & dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) & (V_i > v_threshold) & (V_o > v_threshold)

        f_sponge_cake = (F * D * G) / (4. * dr.abs(cos_theta_i))

        perturb_weight = dr.select(self.perturb_specular, -dr.log(1.0 - self.pcg.next_float32()), 1.0)
        
        weight = (f_sponge_cake / bs.pdf) * perturb_weight
        weight = dr.select(selected_dt, mi.Color3f(1.0, 1.0, 1.0), weight)
        active = active & (dr.all(dr.isfinite(weight)))
        weight = weight & active 

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
        return ('SpongeCake[]')



# class JinSpongeCake (mi.BSDF) : 

#     def __init__ (self, props, *args, texture=None, normal_map=None, 
#             tangent_map=None, perturb_specular=False, delta_transmission=False) : 
#         super().__init__ (props)  
#         self.base_color = props['base_color'] 
#         self.optical_depth = mi.Float(props['optical_depth']) # the product T\rho
#         self.alpha = mi.Float(props['alpha'])
#         self.surface_or_fiber = mi.Bool(props['surface_or_fiber'])
#         self.w = mi.Float(props['w'])

#         reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
#         transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
#         delta_transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
#         self.m_components = [reflection_flags, transmission_flags, delta_transmission_flags]
#         self.m_flags = reflection_flags | transmission_flags | delta_transmission_flags

#         self.pcg = mi.PCG32()

#         self.perturb_specular = mi.Bool(perturb_specular)
#         self.delta_transmission = mi.Bool(delta_transmission)

#         # HACK: force reference so that mitsuba doesn't complain and we can override from cmd line
#         texture_file = props['texture']
#         normal_map_file = props['normal_map']
#         tangent_map_file = props['tangent_map']

#         normal_map_file = normal_map if normal_map is not None else props['normal_map']
#         tangent_map_file = tangent_map if tangent_map is not None else props['tangent_map']
#         texture_file = texture if texture is not None else props['texture']

#         nm = np.array(Image.open(normal_map_file).convert('RGB'))
#         tm = np.array(Image.open(tangent_map_file).convert('RGB'))

#         nm, tm = fix_normal_and_tangent_map(nm, tm)

#         self.normal_map = mi.Texture2f(mi.TensorXf(nm))
#         self.tangent_map = mi.Texture2f(mi.TensorXf(tm))

#         texture_map = np.array(Image.open(texture_file)) / 255.
#         self.texture = mi.Texture2f(mi.TensorXf(texture_map[..., :3]))
#         delta_map = np.ones(tuple(texture_map.shape[:-1]) + (1,)) if not delta_transmission else texture_map[..., 3:]
#         assert delta_map.shape[-1] == 1, "Texture has no or wrong delta transmission channel."
#         # 0 means transmit, 1 means go through
#         self.delta_transmission_map = mi.Texture2f(mi.TensorXf(delta_map))

#     def sample (self, ctx, si, sample1, sample2, active) : 
#         bs = mi.BSDFSample3f() 
#         alpha = dr.maximum(0.0001, self.alpha)
#         S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) # surface type matrix. Later we'll rotate it and all that.
#         S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.)) # fiber type matrix. Later we'll rotate it and all that.

#         normal = mi.Vector3f(self.normal_map.eval(si.uv))
#         tangent = mi.Vector3f(self.tangent_map.eval(si.uv))

#         delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(si.uv))
#         selected_dt = sample1 > delta_transmission.x
        
#         normal = normal / dr.norm(normal)
#         tangent= tangent / dr.norm(tangent)

#         tangent = tangent - dr.sum(tangent * normal) * normal

#         S = dr.select(self.surface_or_fiber, S_surf, S_fibr)

#         S = rotate_s_mat(S, normal, tangent)

#         specular_or_diffuse = sample1 < (delta_transmission.x / 2)

#         h, D, wo, pdf = sample_specular(sample2, si.wi, S)
#         h_, D_, wo_, pdf_ = sample_diffuse_jin(sample2, si.wi)
#         h_dt, D_dt, wo_dt, pdf_dt = mi.Vector3f(0,0,1), 1.0, -si.wi, 1.0

#         h = dr.select(specular_or_diffuse, h, h_)
#         D = dr.select(specular_or_diffuse, D, D_)
#         wo = dr.select(specular_or_diffuse, wo, wo_)
#         pdf = dr.select(specular_or_diffuse, pdf, pdf_)

#         # delta transmission
#         h = dr.select(selected_dt, h_dt, h)
#         D = dr.select(selected_dt, D_dt, D)
#         wo = dr.select(selected_dt, wo_dt, wo)
#         pdf = dr.select(selected_dt, pdf_dt, pdf)

#         color = mi.Color3f(self.texture.eval(si.uv)) 
#         F = color + (1.0 - color) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

#         cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
#         cos_theta_o = mi.Frame3f.cos_theta(wo)

#         selected_r = (cos_theta_i * cos_theta_o > 0.) # reflecting when the two have the same sign

#         G_r = shadow_masking_term_reflect(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)
#         G_t = shadow_masking_term_transmit(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)

#         G = dr.select(selected_r, G_r, G_t)

#         bs.wo = wo
#         bs.eta = 1.
#         bs.sampled_component = dr.select(selected_dt, 
#             mi.UInt32(2), 
#             dr.select(selected_r, mi.UInt32(0), mi.UInt32(1)))
#         bs.sampled_type = dr.select(selected_dt, 
#             mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
#             dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.GlossyTransmission)))
#         bs.pdf = pdf 

#         active = active & dr.neq(cos_theta_i, 0.0) & \
#                 dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) & dr.neq(pdf, 0.0) & \
#                 (specular_or_diffuse | dr.neq(dr.dot(si.wi, normal), 0)) # if we chose the diffuse component,
#                                                                          # please don't let the normal and si.wi
#                                                                          # be parallel.

#         f_sponge_cake = (F * D * G) / (4. * dr.abs(cos_theta_i))
        
#         f_diffuse = (color / dr.pi) * (self.w * dr.abs(cos_theta_i / (dr.dot(si.wi, normal))) + (1 - self.w)) * dr.abs(cos_theta_o)

#         perturb_weight = dr.select(self.perturb_specular, -dr.log(1.0 - self.pcg.next_float32()), 1.0)
        
#         weight = (f_sponge_cake / bs.pdf) * perturb_weight

#         weight = dr.select(selected_dt, mi.Color3f(1.0, 1.0, 1.0), weight)
#         weight = dr.select(specular_or_diffuse, weight, f_diffuse / bs.pdf)

#         active = active & (dr.all(dr.isfinite(weight)))
#         weight = weight & active 
#         return (bs, weight)

#     def eval(self, ctx, si, wo, active):
#         return 0.0

#     def pdf(self, ctx, si, wo, active):
#         return 0.0

#     def eval_pdf(self, ctx, si, wo, active):
#         return 0.0, 0.0

#     def traverse(self, callback):
#         callback.put_parameter('base_color', self.base_color, mi.ParamFlags.Differentiable)
#         callback.put_parameter('optical_depth', self.optical_depth, mi.ParamFlags.Differentiable)
#         callback.put_parameter('alpha', self.alpha, mi.ParamFlags.Differentiable)
#         callback.put_parameter('surface_or_fiber', self.surface_or_fiber, mi.ParamFlags.Discontinuous | mi.ParamFlags.NonDifferentiable)

#     def parameters_changed(self, keys):
#         pass

#     def to_string(self):
#         return ('JinSpongeCake[]')

#     def needs_differentials(self) :
#         return True



class SurfaceBased (mi.BSDF) : 

    def __init__ (self, props, *args, texture=None, normal_map=None, 
            tangent_map=None, perturb_specular=False, delta_transmission=False) : 
        super().__init__ (props)  
        self.base_color = props['base_color'] 
        self.optical_depth = mi.Float(props['optical_depth']) # the product T\rho
        self.alpha = mi.Float(props['alpha'])
        self.surface_or_fiber = mi.Bool(props['surface_or_fiber'])
        self.w = mi.Float(props['w'])

        reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        delta_transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [reflection_flags, transmission_flags, delta_transmission_flags]
        self.m_flags = reflection_flags | transmission_flags | delta_transmission_flags

        self.pcg = mi.PCG32()

        self.perturb_specular = mi.Bool(perturb_specular)
        self.delta_transmission = mi.Bool(delta_transmission)

        # HACK: force reference so that mitsuba doesn't complain and we can override from cmd line
        texture_file = props['texture']
        normal_map_file = props['normal_map']
        tangent_map_file = props['tangent_map']

        normal_map_file = normal_map if normal_map is not None else props['normal_map']
        tangent_map_file = tangent_map if tangent_map is not None else props['tangent_map']
        texture_file = texture if texture is not None else props['texture']

        self.bent_normal_map = mi.Texture2f(mi.TensorXf(fix_map(np.array(Image.open('misc/bent_normal_map.png').convert('RGB')))))
        self.asg_params = mi.Texture2f(mi.TensorXf(np.load('misc/asg_params.npy')))

        nm = np.array(Image.open(normal_map_file).convert('RGB'))
        tm = np.array(Image.open(tangent_map_file).convert('RGB'))

        nm, tm = fix_normal_and_tangent_map(nm, tm)

        self.normal_map = mi.Texture2f(mi.TensorXf(nm))
        self.tangent_map = mi.Texture2f(mi.TensorXf(tm))

        texture_map = np.array(Image.open(texture_file)) / 255.
        self.texture = mi.Texture2f(mi.TensorXf(texture_map[..., :3]))
        delta_map = np.ones(tuple(texture_map.shape[:-1]) + (1,)) if not delta_transmission else texture_map[..., 3:]
        assert delta_map.shape[-1] == 1, "Texture has no or wrong delta transmission channel."
        # 0 means transmit, 1 means go through
        self.delta_transmission_map = mi.Texture2f(mi.TensorXf(delta_map))

    def sample (self, ctx, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 
        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) 
        S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.)) 

        # bent_normal = mi.Vector3f(self.bent_normal_map.eval(si.uv))
        # asg_params = mi.Vector3f(self.asg_params.eval(si.uv))

        normal = mi.Vector3f(self.normal_map.eval(si.uv))
        tangent = mi.Vector3f(self.tangent_map.eval(si.uv))

        delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(si.uv))
        selected_dt = sample1 > delta_transmission.x
        
        normal = normal / dr.norm(normal)
        tangent= tangent / dr.norm(tangent)

        tangent = tangent - dr.sum(tangent * normal) * normal

        ####################################################
        ## Micro-scale BSDF

        ################################
        ## Specular Terms

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)

        S = rotate_s_mat(S, normal, tangent)

        specular_or_diffuse = sample1 < 1.0

        h, D, wo, pdf = sample_specular(sample2, si.wi, S)
        h_, D_, wo_, pdf_ = sample_diffuse(self.pcg, sample2, si.wi, S)
        h_dt, D_dt, wo_dt, pdf_dt = mi.Vector3f(0,0,1), 1.0, -si.wi, 1.0

        h = dr.select(specular_or_diffuse, h, h_)
        D = dr.select(specular_or_diffuse, D, D_)
        wo = dr.select(specular_or_diffuse, wo, wo_)
        pdf = dr.select(specular_or_diffuse, pdf, pdf_)

        # delta transmission
        h = dr.select(selected_dt, h_dt, h)
        D = dr.select(selected_dt, D_dt, D)
        wo = dr.select(selected_dt, wo_dt, wo)
        pdf = dr.select(selected_dt, pdf_dt, pdf)

        color = mi.Color3f(self.texture.eval(si.uv)) 
        F = color + (1.0 - color) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        selected_r = (cos_theta_i * cos_theta_o > 0.) # reflecting when the two have the same sign

        G_r = shadow_masking_term_reflect(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)
        G_t = shadow_masking_term_transmit(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)

        G = dr.select(selected_r, G_r, G_t)

        bs.wo = wo
        bs.eta = 1.
        bs.sampled_component = dr.select(selected_dt, 
            mi.UInt32(2), 
            dr.select(selected_r, mi.UInt32(0), mi.UInt32(1)))
        bs.sampled_type = dr.select(selected_dt, 
            mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
            dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.GlossyTransmission)))
        bs.pdf = pdf 

        # NOTE: removed visibility testing here, please use SpongeCake for that instead
        # EXPERIMENT WITH VISIBILITY
        # di = euclidean_to_spherical_dr(si.wi) 
        # do = euclidean_to_spherical_dr(wo) 
        # bent_normal = bent_normal / dr.norm(bent_normal)
        # mu = euclidean_to_spherical_dr(bent_normal)
        # V_i = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, di)
        # V_o = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, do)

        # v_threshold = 0.5
        active = active & dr.neq(cos_theta_i, 0.0) & dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) # & (V_i > v_threshold) & (V_o > v_threshold)

        f_sponge_cake = (F * D * G) / (4. * dr.abs(cos_theta_i))

        ################################
        ## Diffuse Terms
        
        # NOTE: the previous version (in JinSpongeCake) is
        # f_diffuse = (color / dr.pi) * (self.w * dr.abs(cos_theta_i / (dr.dot(si.wi, normal))) + (1 - self.w)) * dr.abs(cos_theta_o)
        # TODO: check the numerator and denominator, as it is different in the two papers (Jin and SurfaceBased)
        # TODO: now the case of denominator <=0 before clamping is ignored, otherwise it will result in division by 0
        diffuse_sign = dr.select(selected_r, 1.0, -1.0) # negative sign if transmit
        f_diffuse = dr.select(
            diffuse_sign * cos_theta_i <= 0,
            mi.Color3f(0.0, 0.0, 0.0),
            (color / math.pi) * (
                self.w * (clamped_dot(diffuse_sign * si.wi, normal) / clamp_to_nonnegative(diffuse_sign * cos_theta_i)) +
                (1.0 - self.w)
            ) * dr.abs(cos_theta_o)
            # multiplied by the cosine foreshortening factor, as in Mitsuba's documentation:
            # https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF.sample
        )

        s_weight = 0.5 # TODO: think about the weights here; should the two parts add over 1?
        f_surface_based_micro = s_weight * f_sponge_cake + (1.0 - s_weight) * f_diffuse

        ####################################################
        ## Meso-scale BSDF
        ## TODO

        perturb_weight = dr.select(self.perturb_specular, -dr.log(1.0 - self.pcg.next_float32()), 1.0)
        
        weight = (f_surface_based_micro / bs.pdf) * perturb_weight
        weight = dr.select(selected_dt, mi.Color3f(1.0, 1.0, 1.0), weight)
        active = active & (dr.all(dr.isfinite(weight)))
        weight = weight & active 

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
        return ('SurfaceBased[]')
