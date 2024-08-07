

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

    def sample (self, ctx, si, sample1, sample2, active) : 
        tiles = self.tiles
        tiled_uv = dr.select(self.feature_map_type == "cloth", (si.uv * tiles) - dr.trunc(si.uv * tiles), si.uv)
        bs = mi.BSDFSample3f() 
        alpha = dr.maximum(0.0001, self.alpha)
        S_surf = dr.diag(mi.Vector3f(alpha * alpha, alpha * alpha, 1.)) 
        S_fibr = dr.diag(mi.Vector3f(1., alpha * alpha, 1.))

        normal = mi.Vector3f(self.normal_map.eval(tiled_uv))
        tangent = mi.Vector3f(self.tangent_map.eval(tiled_uv))

        delta_transmission = mi.Vector1f(self.delta_transmission_map.eval(tiled_uv))    # TODO: think about the tiled_uv here
        selected_dt = sample1 > delta_transmission.x
        
        normal = normal / dr.norm(normal)
        tangent= tangent / dr.norm(tangent)

        tangent = tangent - dr.sum(tangent * normal) * normal

        ####################################################
        ## Micro-scale BSDF

        specular_or_diffuse = sample1 < self.specular_prob

        ################################
        ## Specular Terms

        S = dr.select(self.surface_or_fiber, S_surf, S_fibr)
        S = rotate_s_mat(S, normal, tangent)

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

        color = mi.Color3f(self.texture.eval(tiled_uv)) # texture usually uses original uv (si.uv) unless using id map
        F = color + (1.0 - color) * ((1 - dr.abs(dr.dot(h, wo))) ** 5)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi) 
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        selected_r = (cos_theta_i * cos_theta_o > 0.0)  # reflecting when the two have the same sign

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

        # VISIBILITY
        bent_normal = mi.Vector3f(self.bent_normal_map.eval(tiled_uv))
        asg_params = mi.Vector3f(self.asg_params.eval(tiled_uv))
        di = euclidean_to_spherical_dr(si.wi) 
        do = euclidean_to_spherical_dr(wo) 
        bent_normal = bent_normal / dr.norm(bent_normal)
        mu = euclidean_to_spherical_dr(bent_normal)
        V_i = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, di)
        V_o = asg_dr(mu, asg_params.x, asg_params.y, asg_params.z, 1.0, do)

        # v_threshold = 0.5
        active = active & dr.neq(cos_theta_i, 0.0) & dr.neq(D, 0.0) & dr.neq(dr.dot(bs.wo, h), 0.0) # & (V_i > v_threshold) & (V_o > v_threshold)

        f_sponge_cake = (F * D * G) / (4. * dr.abs(cos_theta_i))

        ################################
        ## Diffuse Terms

        # TODO: check the numerator and denominator, as it is different in the two papers (Jin and SurfaceBased)
        threshold_diffuse = 0.01    # avoid division of near-zero values
        diffuse_sign = dr.select(selected_r, 1.0, -1.0) # negative sign if transmit
        abs_on = abs_dot(diffuse_sign * si.wi, normal)
        abs_in = dr.abs(diffuse_sign * cos_theta_i)
        abs_in = dr.maximum(threshold_diffuse, abs_in)
        f_diffuse = (F / math.pi) * (
            self.w * (abs_on / abs_in) +
            (1.0 - self.w)
        )

        f_surface_based_micro = dr.select(specular_or_diffuse, f_sponge_cake, f_diffuse)

        ####################################################
        ## Meso-scale BSDF

        ## TODO: ray differentials: texture space partials (si.duv_dx, si.duv_dy) are [0.0, 0.0]
        # si.bsdf()
        # area = dr.norm(dr.cross(si.duv_dx, si.duv_dy))
        ## mipmap level determination for cylinder
        # mipmap_level = dr.round(dr.sqrt(-dr.log2(area)))
        ## TODO: mipmap level determination for cloth, which should consider tiling

        # best threshold value from ASG results for plain type
        v_threshold = 0.15
        V_i = dr.select(V_i <= v_threshold, 0.0, 1.0)
        V_o = dr.select(V_o <= v_threshold, 0.0, 1.0)

        cloth_mipmap_dim = 8
        cylinder_mipmap_dim = 256
        mipmap_level = dr.select(self.feature_map_type == "cloth", int(math.log2(cloth_mipmap_dim)), int(math.log2(cylinder_mipmap_dim)))

        threshold_meso = 0.005              # avoid division of near-zero values
        n_s = mi.Vector3f(0.0, 0.0, 1.0)    # n_s: surface normal
        # n_f: average normal over pixel footprint
        # TODO: interpolation between mipmap levels
        mipmap_level = dr.minimum(mipmap_level, len(self.normal_mipmap)-1)
        n_f = mi.Vector3f(self.normal_mipmap[mipmap_level].eval(tiled_uv))  # TODO: int(mipmap_level.numpy()[0]) if mipmap_level is not yet int
        abs_ns_np = abs_dot(n_s, normal)
        abs_ns_np = dr.maximum(threshold_meso, abs_ns_np)
        abs_ns_nf = abs_dot(n_s, n_f)
        abs_ns_nf = dr.maximum(threshold_meso, abs_ns_nf)
        A_p = (abs_dot(wo, normal)/abs_ns_np) * V_o
        A_g = abs_dot(wo, n_f)/abs_ns_nf
        A_g = dr.maximum(threshold_meso, A_g)
        f_p = f_surface_based_micro * abs_dot(normal, si.wi) * V_i * A_p
        f_surface_based_meso = f_p / A_g    # k_p and pdf cancels out when uniform

        f_overall = f_surface_based_meso
        f_overall *= dr.abs(cos_theta_o)
        # multiplied by the cosine foreshortening factor, as in Mitsuba's documentation:
        # https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF.sample

        perturb_weight = dr.select(self.perturb_specular, -dr.log(1.0 - self.pcg.next_float32()), 1.0)
        
        weight = (f_overall / bs.pdf) * perturb_weight
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
        callback.put_parameter('alpha', self.alpha, mi.ParamFlags.Differentiable)
        callback.put_parameter('optical_depth', self.optical_depth, mi.ParamFlags.Differentiable)
        callback.put_parameter('tiles', self.tiles, mi.ParamFlags.Differentiable)
        callback.put_parameter('specular_prob', self.specular_prob, mi.ParamFlags.Differentiable)
        callback.put_parameter('surface_or_fiber', self.surface_or_fiber, mi.ParamFlags.Discontinuous | mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return ('SurfaceBased[]')
