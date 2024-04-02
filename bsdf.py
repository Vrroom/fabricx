import mitsuba as mi 
import math
import drjit as dr

class TintedDielectricBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.eta = 1.33
        if props.has_property('eta') : 
            self.eta = props['eta']

        self.tint = props['tint']

        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags, transmission_flags] 
        self.m_flags = reflection_flags | transmission_flags

    def sample (self, ctx, si, sample1, sample2, active) : 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, self.eta) 
        t_i = dr.maximum(1.0 - r_i, 0.0)

        selected_r = (sample1 <= r_i) & active 

        bs = mi.BSDFSample3f() 
        bs.pdf = dr.select(selected_r, r_i, t_i)
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1)) 

        bs.sampled_type = dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.DeltaReflection), mi.UInt32(+mi.BSDFFlags.DeltaTransmission))
        bs.wo = dr.select(selected_r,
                          mi.reflect(si.wi),
                          mi.refract(si.wi, cos_theta_t, eta_ti))
        bs.eta = dr.select(selected_r, 1.0, eta_it)

        # value_r = dr.lerp(mi.Color3f(self.tint), mi.Color3f(1.0), dr.clamp(cos_theta_i, 0.0, 1.0))
        value_t = mi.Color3f(1.0) * dr.sqr(eta_ti)
        # value = dr.select(selected_r, value_r, value_t)
        value = dr.select(selected_r, value_t, value_t)
        return (bs, value)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        callback.put_parameter('tint', self.tint, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return ('TintedDielectricBSDF[\n'
                '    eta=%s,\n'
                '    tint=%s,\n'
                ']' % (self.eta, self.tint))

class DisneyDiffuseBSDF (mi.BSDF) : 

    def __init__ (self, props) : 
        super().__init__(props)

        self.base_color = props['base_color']
        self.roughness = props['roughness']
        self.subsurface = props['subsurface'] 

        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags] 
        self.m_flags = reflection_flags 

    def sample (self, ctx, si, sample1, sample2, active) : 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        wo  = mi.warp.square_to_cosine_hemisphere(sample2)
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        h  = (si.wi + wo) / dr.norm(si.wi + wo)

        Fd90    = 0.5 + 2. * self.roughness * (dr.dot(h, wo) ** 2)

        Fd_wi   = (1 + (Fd90 - 1) * (1 - dr.abs(mi.Frame3f.cos_theta(si.wi))) ** 5)
        Fd_wo   = (1 + (Fd90 - 1) * (1 - dr.abs(mi.Frame3f.cos_theta(wo))) ** 5)

        bs = mi.BSDFSample3f() 
        bs.pdf = pdf
        bs.sampled_component = mi.UInt32(0) 

        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.wo = wo
        bs.eta = 1.0 

        n_dot_wi = dr.maximum(0., mi.Frame3f.cos_theta(si.wi))
        n_dot_wo = dr.maximum(0., mi.Frame3f.cos_theta(wo))

        f_base_diffuse = (mi.Color3f(self.base_color) / math.pi) * Fd_wi * Fd_wo * n_dot_wo

        FSS90 = self.roughness * (dr.dot(h, wo) ** 2)
        FSS_wi = (1. + (FSS90 - 1.) * (1 - n_dot_wi) ** 5)
        FSS_wo = (1. + (FSS90 - 1.) * (1 - n_dot_wo) ** 5)

        f_subsurface = ((1.25 * mi.Color3f(self.base_color)) / math.pi) \
                * (FSS_wi * FSS_wo * ((1 / (n_dot_wi + n_dot_wo)) - 0.5) + 0.5) \
                * n_dot_wo

        active = active & dr.neq(n_dot_wi + n_dot_wo, 0.)

        value = ((1 - self.subsurface) * f_base_diffuse + self.subsurface * f_subsurface) / pdf

        return (bs, value & active)

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

        bs = mi.BSDFSample3f() 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        active = active & (cos_theta_i > 0.)

        aspect = dr.sqrt(1 - 0.9 * self.anisotropic)
        alpha_x = dr.maximum(0.0001, self.roughness**2 / aspect)
        alpha_y = dr.maximum(0.0001, self.roughness**2 * aspect)

        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

        m, bs.pdf = distr.sample(si.wi, sample2)

        bs.wo = mi.reflect(si.wi, m)
        bs.eta = 1.
        bs.sampled_component = 0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        active = active & dr.neq(bs.pdf, 0.) & (mi.Frame3f.cos_theta(bs.wo) > 0. )

        Gm = distr.G(si.wi, bs.wo, m)

        Fm = mi.Color3f(self.base_color) + (1 - mi.Color3f(self.base_color)) * (1 - dr.abs(dr.dot(m, bs.wo))) ** 5 

        weight = Gm * dr.dot(bs.wo, m) / cos_theta_i

        bs.pdf /= 4. * dr.dot(bs.wo, m)

        weight *= Fm

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
        return ('DisneyMetalBSDF[]')
