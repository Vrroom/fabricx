import mitsuba as mi 
import drjit as dr

def half_vec (wi, wo) :
    return  (wi + wo) / dr.norm(wi + wo)

def safe_rsqrt (x) : 
    return dr.rsqrt(dr.maximum(x, 0.))

def sggx_sample(sh_frame, sample, s_mat) :
    k, j, i = 0, 1, 2
    m = mi.Matrix3f (sh_frame.s, sh_frame.t, sh_frame.n)
    m = dr.transpose(m);
    s2 = m @ s_mat @ dr.transpose(m)
    inv_sqrt_s_ii = safe_rsqrt(s2[i, i])
    tmp = dr.safe_sqrt(s2[j, j] * s2[i, i] - s2[j, i] * s2[j, i])
    m_k = mi.Vector3f(dr.safe_sqrt(dr.abs(dr.det(s2))) / tmp, 0., 0.);
    m_j = mi.Vector3f(-inv_sqrt_s_ii * (s2[k, i] * s2[j, i] - s2[k, j] * s2[i, i]) / tmp, inv_sqrt_s_ii * tmp, 0.)
    m_i = inv_sqrt_s_ii * mi.Vector3f(s2[k, i], s2[j, i], s2[i, i])
    uvw = mi.warp.square_to_cosine_hemisphere(sample)
    return sh_frame.to_world(dr.normalize(uvw.x * m_k + uvw.y * m_j + uvw.z * m_i))

def abs_det_3_by_3 (s) : 
    """ compute the determinant of three by three matrix using dr jit ops """ 
    return dr.abs(
        s[0, 0] * (s[1, 1] * s[2, 2] - s[1, 2] * s[2, 1]) \
        -s[0, 1] * (s[1, 0] * s[2, 2] - s[1, 2] * s[2, 0]) \
        +s[0, 2] * (s[1, 0] * s[2, 1] - s[1, 1] * s[2, 0]))

def their_abs_det (s) : 
    """ same as what I wrote, atleast for symmetric matrices """ 
    return dr.abs(s[0, 0] * s[1, 1] * s[2, 2] - s[0, 0] * s[1, 2] * s[1, 2] -
                          s[1, 1] * s[0, 2] * s[0, 2] - s[2, 2] * s[0, 1] * s[0, 1] +
                          2. * s[0, 1] * s[0, 2] * s[1, 2]);

def sggx_pdf(wm, s_mat) : 
    det_s = abs_det_3_by_3(s_mat)
    den = wm.x * wm.x * (s_mat[1, 1] * s_mat[2, 2] - s_mat[1, 2] * s_mat[1, 2]) + \
          wm.y * wm.y * (s_mat[0, 0] * s_mat[2, 2] - s_mat[0, 2] * s_mat[0, 2]) + \
          wm.z * wm.z * (s_mat[0, 0] * s_mat[1, 1] - s_mat[0, 1] * s_mat[0, 1]) + \
          2. * (wm.x * wm.y * (s_mat[0, 2] * s_mat[1, 2] - s_mat[2, 2] * s_mat[0, 1]) + \
                wm.x * wm.z * (s_mat[0, 1] * s_mat[1, 2] - s_mat[1, 1] * s_mat[0, 2]) + \
                wm.y * wm.z * (s_mat[0, 1] * s_mat[0, 2] - s_mat[0, 0] * s_mat[1, 2]))
    return dr.maximum(det_s, 0.) * dr.safe_sqrt(det_s) \
           / (dr.pi * dr.sqr(den))

def sggx_projected_area(wi, s_mat) : 
    sigma2 = wi.x * wi.x * s[0, 0] + wi.y * wi.y * s[1, 1] + \
             wi.z * wi.z * s[2, 2] + \
             2. * (wi.x * wi.y * s[0, 1] + wi.x * wi.z * s[0, 2] + \
                   wi.y * wi.z * s[1, 2])
    return dr.safe_sqrt(sigma2)

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

def diffuse_lobe_sample (wi, sample1, sample2, *args, **kwargs) : 
    wo  = mi.warp.square_to_cosine_hemisphere(sample2)
    pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
    return wo, pdf, True, mi.UInt32(+mi.BSDFFlags.DiffuseReflection), dict()

class SolidTextureBSDF (mi.BSDF) : 

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
        
        s_mat = mi.Matrix3f([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p_val = dr.clamp(2 * sggx_pdf(si.p, s_mat), 0, 1)
        base_color = mi.Color3f(mi.Vector3f(p_val, p_val, 0.))
        f_diffuse = diffuse_bsdf(si.wi, wo, cos_theta_i, cos_theta_o, self.roughness, base_color, self.subsurface)

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
        print()

    def to_string(self):
        return ('SolidTextureBSDF[]')

