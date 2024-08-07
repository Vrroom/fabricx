from PIL import Image
import numpy as np
import math
import os
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

class SurfaceBased (mi.BSDF) : 

    def __init__ (self, props, *args, feature_map_type=None, cloth_type=None, texture=None, perturb_specular=False, delta_transmission=False) : 
        super().__init__ (props)  
        self.base_color = props['base_color'] 
        self.optical_depth = mi.Float(props['optical_depth']) # the product T\rho
        self.alpha = mi.Float(props['alpha'])
        self.tiles = mi.Float(props['tiles'])
        self.specular_prob = mi.Float(props['specular_prob'])
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

        self.feature_map_type = feature_map_type    # default "cloth"
        self.cloth_type = cloth_type                # default "plain"

        normal_map_file = "normal_map.png"
        tangent_map_file = "tangent_map.png"
        texture_file = texture                      # default "id_map.png"

        if self.feature_map_type == "cloth":
            feature_map_dir = os.path.join(self.feature_map_type, self.cloth_type)
        else:
            feature_map_dir = self.feature_map_type
        
        normal_map_path: str = os.path.join(feature_map_dir, normal_map_file)
        tangent_map_path: str = os.path.join(feature_map_dir, tangent_map_file)
        texture_path: str = os.path.join(feature_map_dir, texture_file)

        # Reading Normal Map
        nm = None
        if (normal_map_path.endswith(".png")):
            nm = np.array(Image.open(normal_map_path).convert("RGB"), dtype=float)
            nm /= 255.0
        elif (normal_map_path.endswith(".txt")):
            nm = read_txt_feature_map(normal_map_path)
        else:
            raise NotImplementedError("Normal map file must be .png or .txt")

        # Reading Tangent Map
        tm = None
        if (tangent_map_path.endswith(".png")):
            tm = np.array(Image.open(tangent_map_path).convert("RGB"), dtype=float)
            tm /= 255.0
        elif (tangent_map_path.endswith(".txt")):
            tm = read_txt_feature_map(tangent_map_path)

        nm, tm = fix_normal_and_tangent_map(nm, tm)

        self.normal_map = mi.Texture2f(mi.TensorXf(nm))
        self.tangent_map = mi.Texture2f(mi.TensorXf(tm))

        # Reading Texture Map
        texture_map = None
        if (texture_path.endswith(".png")):
            texture_map = np.array(Image.open(texture_path)) / 255.0
        elif (texture_path.endswith(".txt")):
            texture_map = read_txt_feature_map(texture_path, 4 if self.delta_transmission else 3)
        self.texture = mi.Texture2f(mi.TensorXf(texture_map[..., :3]))
        delta_map = np.ones(tuple(texture_map.shape[:-1]) + (1,)) if not delta_transmission else texture_map[..., 3:]
        assert delta_map.shape[-1] == 1, "Texture has no or wrong delta transmission channel."
        # 0 means transmit, 1 means go through
        self.delta_transmission_map = mi.Texture2f(mi.TensorXf(delta_map))

        # Reading Visibility Maps
        self.bent_normal_map = mi.Texture2f(mi.TensorXf(fix_map(np.array(Image.open(os.path.join(feature_map_dir, "bent_normal_map.png")).convert("RGB")))))
        self.asg_params = mi.Texture2f(mi.TensorXf(np.load(os.path.join(feature_map_dir, "asg_params.npy"))))

        # Reading Normal Mipmaps
        self.normal_mipmap = []
        feature_map_dim = nm.shape[0]
        cur_dim = 1
        while cur_dim < feature_map_dim:
            mipmap_path = os.path.join(feature_map_dir, "normal_" + str(cur_dim) + ".png")
            self.normal_mipmap.append(mi.Texture2f(mi.TensorXf(fix_map(np.array(Image.open(mipmap_path).convert("RGB"))))))
            cur_dim *= 2
        self.normal_mipmap.append(self.normal_map)

    
    def sample(self, ctx, si, sample1, sample2, active): 
        tiles = self.tiles
        tiled_uv = dr.select(self.feature_map_type == "cloth", (si.uv * tiles) - dr.trunc(si.uv * tiles), si.uv)
        color = mi.Color3f(self.texture.eval(tiled_uv)) # four different color parameters? (rd, rs, td, ts)
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(tiled_uv))    # TODO: think about the tiled_uv here
        delta_val = dr.select(delta_transmission.x > 1.0, 1.0, delta_transmission.x)
        delta_val = dr.select(delta_val<0.0, 0.0, delta_val)
        selected_dt = sample1 > delta_val

        bs = mi.BSDFSample3f()
        reflect_or_transmit = sample1 < 0.5 #(TODO) probably good to decorrelate this with delta transmission check

        f_diffuse = color / math.pi        
        
        wo_diffuse = mi.warp.square_to_cosine_hemisphere(sample2)
        pdf_diffuse = 0.5 * mi.warp.square_to_cosine_hemisphere_pdf(wo_diffuse)
        wo_diffuse = dr.select(reflect_or_transmit, wo_diffuse, -wo_diffuse)
        cos_theta_o_diffuse = mi.Frame3f.cos_theta(wo_diffuse)
        
        # specular sample
        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) 
        S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.))
        # S_surf = dr.diag(dr.llvm.ad.Array3f(alpha * alpha, alpha * alpha, 1.)) # surface type matrix. Later we'll rotate it and all that.
        # S_fibr = dr.diag(dr.llvm.ad.Array3f(1., alpha * alpha, 1.)) # fiber type matrix. Later we'll rotate it and all that.

        normal = mi.Vector3f(self.normal_map.eval(tiled_uv))
        tangent = mi.Vector3f(self.tangent_map.eval(tiled_uv))        
        normal = normal / dr.norm(normal)
        tangent= tangent / dr.norm(tangent)
        tangent = tangent - dr.sum(tangent * normal) * normal

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)
        S = rotate_s_mat(S, normal, tangent)

        #h, D, wo_spec, pdf_spec = sample_specular(sample2, si.wi, S)
        h = dr.normalize(si.wi + wo_diffuse)
        D = sggx_pdf(h, S)
        D = dr.select(dr.isnan(D), 0.0, D)

        G_r = shadow_masking_term_reflect(si.wi, wo_diffuse, cos_theta_i, cos_theta_o_diffuse, self.optical_depth, S)
        G_t = shadow_masking_term_transmit(si.wi, wo_diffuse, cos_theta_i, cos_theta_o_diffuse, self.optical_depth, S)        

        G = dr.select(reflect_or_transmit, G_r, G_t)
        G = dr.select(dr.isnan(G), 0.0, G)

        # VISIBILITY
        bent_normal = mi.Vector3f(self.bent_normal_map.eval(tiled_uv))
        asg_params = mi.Vector3f(self.asg_params.eval(tiled_uv))
        di = euclidean_to_spherical_dr(si.wi) 
        do = euclidean_to_spherical_dr(wo_diffuse) 
        bent_normal = bent_normal / dr.norm(bent_normal)
        mu = euclidean_to_spherical_dr(bent_normal)
        V_i = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, di)
        V_o = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, do)

        f_sponge_cake = (color * D * G) / (4. * dr.abs(cos_theta_i))

        f_surface_based_micro = self.specular_prob * f_sponge_cake + \
                                (1-self.specular_prob) * f_diffuse

        ###################################################
        # Meso-scale BSDF
        v_threshold = 0.15
        V_i = dr.select(V_i <= v_threshold, 0.0, 1.0)
        V_o = dr.select(V_o <= v_threshold, 0.0, 1.0)
        f_surface_based_meso = f_surface_based_micro * V_i * V_o

        ####### final values
        wo = dr.select(selected_dt, -si.wi, wo_diffuse)
        #pdf = dr.select(selected_dt, 1.0, pdf_diffuse)
        pdf = dr.select(selected_dt, 1.0, (1-delta_val) * pdf_diffuse)        

        bs.sampled_component = dr.select(selected_dt, 
            mi.UInt32(2), 
             dr.select(reflect_or_transmit, mi.UInt32(0), mi.UInt32(1)))
        bs.sampled_type = dr.select(selected_dt, 
            mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
            dr.select(reflect_or_transmit, mi.UInt32(+mi.BSDFFlags.DiffuseReflection),
                                    mi.UInt32(+mi.BSDFFlags.DiffuseTransmission)))
       
        bs.wo = wo
        bs.eta = 1.       
        active = active & dr.neq(cos_theta_i * cos_theta_o_diffuse, 0.0) & (pdf > 0.0)
        
        weight = dr.select(selected_dt, 1.0, f_surface_based_meso * dr.abs(cos_theta_o_diffuse) / pdf)
        active = active & (dr.all(dr.isfinite(weight)))
        weight = dr.select(active, weight, mi.Color3f(0.0,0.0,0.0))      
        pdf = dr.select(active, pdf, 0.0)
        bs.pdf = pdf

        return (bs, weight)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        tiles = self.tiles
        tiled_uv = dr.select(self.feature_map_type == "cloth", (si.uv * tiles) - dr.trunc(si.uv * tiles), si.uv)
        color = mi.Color3f(self.texture.eval(tiled_uv))

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        f_diffuse = color / math.pi 

        wo_positive = dr.select(wo.z>0, wo, mi.Vector3f(wo.x, wo.y, -wo.z))
        pdf_diffuse = 0.5 * mi.warp.square_to_cosine_hemisphere_pdf(wo_positive)        

        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) 
        S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.))
        # S_surf = dr.diag(dr.llvm.ad.Array3f(alpha * alpha, alpha * alpha, 1.)) # surface type matrix. Later we'll rotate it and all that.
        # S_fibr = dr.diag(dr.llvm.ad.Array3f(1., alpha * alpha, 1.)) # fiber type matrix. Later we'll rotate it and all that.

        normal = mi.Vector3f(self.normal_map.eval(tiled_uv))
        tangent = mi.Vector3f(self.tangent_map.eval(tiled_uv))
        # if eval then it shouldn't be delta transmission because the probability
        # of picking the continuing direction is 0?        
        normal = normal / dr.norm(normal)
        tangent= tangent / dr.norm(tangent)

        tangent = tangent - dr.sum(tangent * normal) * normal

        delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(tiled_uv))

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)
        S = rotate_s_mat(S, normal, tangent)

        h = dr.normalize(si.wi + wo)
        D = sggx_pdf(h, S)
        D = dr.select(dr.isnan(D), 0.0, D)

        selected_r = (cos_theta_i * cos_theta_o > 0.0)  # reflecting when the two have the same sign
        G_r = shadow_masking_term_reflect(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)
        G_t = shadow_masking_term_transmit(si.wi, wo, cos_theta_i, cos_theta_o, self.optical_depth, S)       


        # Mandy hacks now so transmission is 0 energy
        G = dr.select(selected_r, G_r, G_t)
        G = dr.select(dr.isnan(G), 0.0, G)

        # VISIBILITY
        bent_normal = mi.Vector3f(self.bent_normal_map.eval(tiled_uv))
        asg_params = mi.Vector3f(self.asg_params.eval(tiled_uv))
        di = euclidean_to_spherical_dr(si.wi) 
        do = euclidean_to_spherical_dr(wo) 
        bent_normal = bent_normal / dr.norm(bent_normal)
        mu = euclidean_to_spherical_dr(bent_normal)
        V_i = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, di)
        V_o = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, do)
        V_i = dr.select(dr.isnan(V_i), 0.0, V_i)
        V_o = dr.select(dr.isnan(V_o), 0.0, V_o)

        #delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(tiled_uv))    # TODO: think about the tiled_uv here
        pdf = pdf_diffuse * (1-delta_transmission.x)
        f_sponge_cake = (color * D * G) / (4. * dr.abs(cos_theta_i))

        f_surface_based_micro = self.specular_prob * f_sponge_cake + \
                        (1 - self.specular_prob) * f_diffuse

        ###################################################
        # Meso-scale BSDF
        f_surface_based_meso = f_surface_based_micro * V_i * V_o
        value =  f_surface_based_meso * dr.abs(cos_theta_o)


        active = active & dr.neq(D, 0.0) & dr.neq(dr.dot(wo, h), 0.0)
        # active =  active & (cos_theta_i != 0.0 ) & (D != 0.0) & (dr.dot(bs.wo, h) != 0.0)        
        active = active & dr.neq(cos_theta_i * cos_theta_o, 0.0) & (pdf_diffuse > 0.0)
        pdf = dr.select(active, pdf, 0.0)
        value = dr.select(active, value, mi.Color3f(0.0,0.0,0.0))


        # hack diffuse only
        # value = f_diffuse * cos_theta_o
        # pdf = pdf_diffuse

        return value, pdf


    def traverse(self, callback):
        callback.put_parameter('base_color', self.base_color, mi.ParamFlags.Differentiable)
        callback.put_parameter('alpha', self.alpha, mi.ParamFlags.Differentiable)
        callback.put_parameter('optical_depth', self.optical_depth, mi.ParamFlags.Differentiable)
        callback.put_parameter('tiles', self.tiles, mi.ParamFlags.Differentiable)
        callback.put_parameter('specular_prob', self.specular_prob, mi.ParamFlags.Differentiable)
        callback.put_parameter('surface_or_fiber', self.surface_or_fiber, mi.ParamFlags.Discontinuous | mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return ('SurfaceBased[]')
