import mitsuba as mi
import drjit as dr

def half_vec (wi, wo) : 
    return (wi + wo) / dr.norm(wi + wo)

def test_det_3_by_3  () :
    import numpy as np
    for i in range(10): 
        mat = np.random.randn(3, 3)
        mat = mat + mat.T
        s_mat = mi.Matrix3f(mat)
        np_det = float(np.abs(np.linalg.det(mat)))
        mi_det = float(abs_det_3_by_3(s_mat)[0])
        mi_det2 = float(their_abs_det(s_mat)[0])
        assert abs(np_det - mi_det) < 1e-5, f'Difference between numpy and mi is = {abs(np_det - mi_det):.5f}'
        assert abs(np_det - mi_det2) < 1e-5, f'Difference between numpy and their mi is = {abs(np_det - mi_det):.5f}'

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
    """
    This is an even function in wi. That is sggx_projected_area(wi, s_mat) = sggx_projected_area(-wi, s_mat)
    """
    sigma2 = wi.x * wi.x * s_mat[0, 0] + wi.y * wi.y * s_mat[1, 1] + \
             wi.z * wi.z * s_mat[2, 2] + \
             2. * (wi.x * wi.y * s_mat[0, 1] + wi.x * wi.z * s_mat[0, 2] + \
                   wi.y * wi.z * s_mat[1, 2])
    return dr.safe_sqrt(sigma2)

if __name__ == "__main__" : 
    # TODO: Plot and verify the distribution functions from the original SGGX microflake paper
    mi.set_variant('llvm_ad_rgb') 
    sh_frame = mi.Frame3f([1,0,0])
    sample = mi.Point2f([0.2, 0.1])
    s_mat = mi.Matrix3f([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    test_det_3_by_3()
    print(abs_det_3_by_3(s_mat))
    print(sggx_sample(sh_frame, sample, s_mat))
    print(sggx_pdf(sggx_sample(sh_frame, sample, s_mat), s_mat))
