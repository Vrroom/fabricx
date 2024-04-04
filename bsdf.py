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

        reflection_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

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

        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
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
        bs = mi.BSDFSample3f() 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active = active & dr.neq(cos_theta_i, 0.) # now since we are transmitting, a ray can hit from inside

        aspect = dr.sqrt(1 - 0.9 * self.anisotropic)
        alpha_x = dr.maximum(0.0001, self.roughness**2 / aspect)
        alpha_y = dr.maximum(0.0001, self.roughness**2 * aspect)

        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

        m, bs.pdf = distr.sample(dr.mulsign(si.wi, cos_theta_i), sample2)
        active &= dr.neq(bs.pdf, 0.)

        F, cos_theta_t, eta_it, eta_ti = mi.fresnel(dr.dot(si.wi, m), self.eta)

        selected_r = (sample1 <= F) & active

        weight = 1.
        bs.pdf *= dr.select(selected_r, F, 1. - F)

        # selected_t = ~selected_r & active
        bs.eta = dr.select(selected_r, 1., eta_it);
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1)) 
        bs.sampled_type = dr.select(selected_r, 
            mi.UInt32(+mi.BSDFFlags.GlossyReflection), 
            mi.UInt32(+mi.BSDFFlags.GlossyTransmission)
        )

        dwh_dwo = 0.

        bs.wo = dr.select(selected_r, mi.reflect(si.wi, m), mi.refract(si.wi, m, cos_theta_t, eta_ti))
        dwh_dwo = dr.select(selected_r, 
            1. / (4. * dr.dot(bs.wo, m)),
            (dr.sqr(bs.eta) * dr.dot(bs.wo, m)) / dr.sqr(dr.dot(si.wi, m) + bs.eta * dr.dot(bs.wo, m))
        )
        base_color = mi.Color3f(self.base_color)

        actual_color = dr.select(selected_r, base_color, dr.sqrt(base_color))

        weight *= actual_color * distr.G(si.wi, bs.wo, m) * dr.dot(si.wi, m) / (cos_theta_i) 
        weight *= dr.select(selected_r, 1., 1. / dr.sqr(bs.eta))
        bs.pdf *= dr.abs(dwh_dwo)
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

        alpha_g = (1. - self.clearcoat_gloss) * 0.1 + self.clearcoat_gloss * 0.001
        alpha_g_sq = alpha_g ** 2

        cos_h_el = dr.sqrt((1. - dr.power(alpha_g_sq, 1 - sample2[0])) / (1 - alpha_g_sq))
        sin_h_el = dr.sqrt(1.0 - cos_h_el ** 2)

        h_az = 2 * math.pi * sample2[1]

        hx = sin_h_el * dr.cos(h_az)
        hy = sin_h_el * dr.sin(h_az)
        hz = cos_h_el

        h = mi.Vector3f(hx, hy, hz) 

        wo = mi.reflect(si.wi, h) 
        
        R0 = (1.5 - 1) ** 2 / (1.5 + 1) ** 2

        Fc = R0 + (1 - R0) * (1. - dr.abs(dr.dot(h, wo))) ** 5

        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, 0.25, 0.25)

        Gc = distr.G(si.wi, wo, h)

        Dc = (alpha_g_sq - 1.) / (math.pi * dr.log(alpha_g_sq) * (1. + (alpha_g_sq - 1) * (hz ** 2))) 

        bs.wo = wo
        bs.eta = 1.
        bs.pdf = Dc / (4. * dr.dot(bs.wo, h))
        bs.sampled_component = 0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        active = active & dr.neq(bs.pdf, 0.) & (mi.Frame3f.cos_theta(bs.wo) > 0. )

        f_clearcoat = Fc * Dc * Gc / (4 * dr.abs(cos_theta_i)) 

        weight = f_clearcoat / bs.pdf

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
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        bs = mi.BSDFSample3f() 

        wo  = mi.warp.square_to_cosine_hemisphere(sample2)
        cos_theta_o = mi.Frame3f.cos_theta(wo) 
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        base_color = mi.Color3f(self.base_color)
        white_color = mi.Color3f([1., 1., 1.])

        lum = mi.luminance(base_color)

        C_tint = dr.select(lum > 0, base_color / lum, white_color)
        C_sheen = (1 - self.sheen_tint) + self.sheen_tint * C_tint

        h  = (si.wi + wo) / dr.norm(si.wi + wo)

        f_sheen = C_sheen * ((1 - dr.abs(dr.dot(h, wo))) ** 5) * dr.abs(wo)

        active = active & (cos_theta_i > 0.)

        bs.wo = wo
        bs.eta = 1.
        bs.pdf = pdf
        bs.sampled_component = 0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        active = active & dr.neq(bs.pdf, 0.) & (cos_theta_o > 0. )

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

        print(self.diffuseWeight, self.sheenWeight, self.metalWeight)
        self.total = sum([self.diffuseWeight, self.sheenWeight, self.metalWeight]) #, self.glassWeight, self.clearcoatWeight]) 

        self.diffuseProb = self.diffuseWeight / self.total
        self.sheenProb = self.sheenWeight / self.total
        self.metalProb = self.metalWeight / self.total
        # self.glassProb = self.glassWeight / self.total
        # self.clearcoatProb = self.clearcoatWeight / self.total

        diffuse_reflection_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        glossy_reflection_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [diffuse_reflection_flags, glossy_reflection_flags] 
        self.m_flags = diffuse_reflection_flags | glossy_reflection_flags


    def f_sheen (self, si, sample1, sample2, active) : 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        bs = mi.BSDFSample3f() 

        wo  = mi.warp.square_to_cosine_hemisphere(sample2)
        cos_theta_o = mi.Frame3f.cos_theta(wo) 
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        base_color = mi.Color3f(self.base_color)
        white_color = mi.Color3f([1., 1., 1.])

        lum = mi.luminance(base_color)

        C_tint = dr.select(lum > 0, base_color / lum, white_color)
        C_sheen = (1 - self.sheen_tint) + self.sheen_tint * C_tint

        h  = (si.wi + wo) / dr.norm(si.wi + wo)

        f_sheen = C_sheen * ((1 - dr.abs(dr.dot(h, wo))) ** 5) * dr.abs(wo)

        is_active = active & (cos_theta_i > 0.)

        bs.wo = wo
        bs.eta = 1.
        bs.pdf = pdf
        bs.sampled_component = 0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        is_active = is_active & dr.neq(bs.pdf, 0.) & (cos_theta_o > 0. )

        weight = f_sheen / bs.pdf

        return (f_sheen, bs.pdf, is_active, bs.wo, h)

    #def f_clearcoat (self, si, sample1, sample2, active) : 
    #    bs = mi.BSDFSample3f() 
    #    cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
    #    is_active = active & (cos_theta_i > 0.)

    #    alpha_g = (1. - self.clearcoat_gloss) * 0.1 + self.clearcoat_gloss * 0.001
    #    alpha_g_sq = alpha_g ** 2

    #    cos_h_el = dr.sqrt((1. - dr.power(alpha_g_sq, 1 - sample2[0])) / (1 - alpha_g_sq))
    #    sin_h_el = dr.sqrt(1.0 - cos_h_el ** 2)

    #    h_az = 2 * math.pi * sample2[1]

    #    hx = sin_h_el * dr.cos(h_az)
    #    hy = sin_h_el * dr.sin(h_az)
    #    hz = cos_h_el

    #    h = mi.Vector3f(hx, hy, hz) 

    #    wo = mi.reflect(si.wi, h) 
    #    
    #    R0 = (1.5 - 1) ** 2 / (1.5 + 1) ** 2

    #    Fc = R0 + (1 - R0) * (1. - dr.abs(dr.dot(h, wo))) ** 5

    #    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, 0.25, 0.25)

    #    Gc = distr.G(si.wi, wo, h)

    #    Dc = (alpha_g_sq - 1.) / (math.pi * dr.log(alpha_g_sq) * (1. + (alpha_g_sq - 1) * (hz ** 2))) 

    #    bs.wo = wo
    #    bs.eta = 1.
    #    bs.pdf = Dc / (4. * dr.dot(bs.wo, h))
    #    bs.sampled_component = 0
    #    bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

    #    is_active = is_active & dr.neq(bs.pdf, 0.) & (mi.Frame3f.cos_theta(bs.wo) > 0. )

    #    f_clearcoat = Fc * Dc * Gc / (4 * dr.abs(cos_theta_i)) 

    #    return (bs, f_clearcoat, bs.pdf, is_active)

    #def f_glass (self, si, sample1, sample2, active) : 
    #    bs = mi.BSDFSample3f() 
    #    cos_theta_i = mi.Frame3f.cos_theta(si.wi)
    #    is_active = active & dr.neq(cos_theta_i, 0.) # now since we are transmitting, a ray can hit from inside

    #    aspect = dr.sqrt(1 - 0.9 * self.anisotropic)
    #    alpha_x = dr.maximum(0.0001, self.roughness**2 / aspect)
    #    alpha_y = dr.maximum(0.0001, self.roughness**2 * aspect)

    #    distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

    #    m, bs.pdf = distr.sample(dr.mulsign(si.wi, cos_theta_i), sample2)
    #    is_active = is_active & dr.neq(bs.pdf, 0.)

    #    F, cos_theta_t, eta_it, eta_ti = mi.fresnel(dr.dot(si.wi, m), self.eta)

    #    selected_r = (sample1 <= F) & is_active

    #    weight = 1.
    #    bs.pdf *= dr.select(selected_r, F, 1. - F)

    #    bs.eta = dr.select(selected_r, 1., eta_it);
    #    bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1)) 
    #    bs.sampled_type = dr.select(selected_r, 
    #        mi.UInt32(+mi.BSDFFlags.GlossyReflection), 
    #        mi.UInt32(+mi.BSDFFlags.GlossyTransmission)
    #    )

    #    dwh_dwo = 0.

    #    bs.wo = dr.select(selected_r, mi.reflect(si.wi, m), mi.refract(si.wi, m, cos_theta_t, eta_ti))
    #    dwh_dwo = dr.select(selected_r, 
    #        1. / (4. * dr.dot(bs.wo, m)),
    #        (dr.sqr(bs.eta) * dr.dot(bs.wo, m)) / dr.sqr(dr.dot(si.wi, m) + bs.eta * dr.dot(bs.wo, m))
    #    )
    #    base_color = mi.Color3f(self.base_color)

    #    actual_color = dr.select(selected_r, base_color, dr.sqrt(base_color))

    #    weight *= actual_color * distr.G(si.wi, bs.wo, m) * dr.dot(si.wi, m) / (cos_theta_i) 
    #    weight *= dr.select(selected_r, 1., 1. / dr.sqr(bs.eta))
    #    bs.pdf *= dr.abs(dwh_dwo)
    #    return (bs, weight * bs.pdf, bs.pdf, is_active)

    def f_diffuse (self, si, sample1, sample2, active) : 
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

        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
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

        is_active = active & dr.neq(n_dot_wi + n_dot_wo, 0.)

        f_diffuse = ((1 - self.subsurface) * f_base_diffuse + self.subsurface * f_subsurface) 

        return (f_diffuse, pdf, is_active, wo, h)

    def f_metal (self, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        is_active = active & (cos_theta_i > 0.)

        aspect = dr.sqrt(1 - 0.9 * self.anisotropic)
        alpha_x = dr.maximum(0.0001, self.roughness**2 / aspect)
        alpha_y = dr.maximum(0.0001, self.roughness**2 * aspect)

        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_x, alpha_y)

        m, bs.pdf = distr.sample(si.wi, sample2)

        bs.wo = mi.reflect(si.wi, m)
        bs.eta = 1.
        bs.sampled_component = 0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        is_active = is_active & dr.neq(bs.pdf, 0.) & (mi.Frame3f.cos_theta(bs.wo) > 0. )

        Gm = distr.G(si.wi, bs.wo, m)

        base_color = mi.Color3f(self.base_color)
        white_color = mi.Color3f([1., 1., 1.])

        lum = mi.luminance(base_color)

        C_tint = dr.select(lum > 0, base_color / lum, white_color)
        Ks = (1 - self.specular_tint) + self.specular_tint * C_tint
        R0 = (1.5 - 1) ** 2 / (1.5 + 1) ** 2
        C0 = self.specular * R0 * (1 - self.metallic) * Ks + self.metallic * base_color

        Fm = C0 + (1 - C0) * (1 - dr.abs(dr.dot(m, bs.wo))) ** 5 

        weight = Gm * dr.dot(bs.wo, m) / cos_theta_i

        bs.pdf /= 4. * dr.dot(bs.wo, m)

        weight *= Fm

        return ((weight * bs.pdf), bs.pdf, is_active, bs.wo, m)

    def sample (self, ctx, si, sample1, sample2, active) : 
        bs = mi.BSDFSample3f() 
        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 

        f_diffuse, pdf_diffuse, active_diffuse, wo_diffuse, m_diffuse = self.f_diffuse(si, sample1, sample2, active)
        f_sheen, pdf_sheen, active_sheen, wo_sheen, m_sheen = self.f_sheen(si, sample1, sample2, active)
        f_metal, pdf_metal, active_metal, wo_metal, m_metal = self.f_metal(si, sample1, sample2, active)
        # bs_clearcoat, f_clearcoat, pdf_clearcoat, active_clearcoat = self.f_clearcoat(si, sample1, sample2, active)
        # bs_glass, f_glass, pdf_glass, active_glass = self.f_glass(si, sample1, sample2, active)
                
        # f_diffuse = dr.select(cos_theta_i <= 0, 0, f_diffuse)
        # f_metal = dr.select(cos_theta_i <= 0, 0, f_metal)

        #f_clearcoat = dr.select(cos_theta_i <= 0, 0, f_clearcoat)
        #f_sheen = dr.select(cos_theta_i <= 0, 0, f_sheen)

        # f_disney = self.diffuseWeight * f_diffuse \
        #         + (1 - self.metallic) * self.sheen * f_sheen \
        #         + self.metalWeight * f_metal \
        #         + self.clearcoatWeight * f_clearcoat \
        #         + self.glassWeight * f_glass
        f_disney = dr.select(
            sample1 <= self.diffuseProb, 
            f_diffuse, 
            dr.select(
                sample1 <= self.diffuseProb + self.sheenProb,
                f_sheen, 
                f_metal
            )
        )
        # f_disney = (self.diffuseWeight * f_diffuse) \
        #         + (self.metalWeight * f_metal)

        bs.wo = dr.select(
            sample1 <= self.diffuseProb, 
            wo_diffuse, 
            dr.select(
                sample1 <= self.diffuseProb + self.sheenProb, 
                wo_sheen,
                wo_metal
            )
        )

        bs.eta = 1.

        bs.sampled_component = dr.select(
            sample1 <= self.diffuseProb, 
            mi.UInt32(0), 
            mi.UInt32(1)
        )

        bs.sampled_type = dr.select(
            sample1 <= self.diffuseProb, 
            mi.UInt32(+mi.BSDFFlags.DiffuseReflection), 
            mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        )

        P = dr.select(
            sample1 <= self.diffuseProb,
            self.diffuseProb,
            dr.select(
                sample1 <= self.diffuseProb + self.sheenProb,
                self.sheenProb,
                self.metalProb
            )
        )

        bs.pdf = dr.select(
            sample1 <= self.diffuseProb, 
            pdf_diffuse, 
            dr.select(
                sample1 <= self.diffuseProb + self.sheenProb, 
                pdf_sheen,
                pdf_metal
            ) 
        )

        bs.pdf *= P

        active = dr.select(sample1 <= self.diffuseProb, 
            active_diffuse, 
            dr.select(
                sample1 <= self.diffuseProb + self.sheenProb,
                active_sheen,
                active_metal
            )
        )
        
        return (bs, (f_disney / bs.pdf) & active)

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
        Return ('DisneyPrincipledBSDF[]')
