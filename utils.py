import mitsuba as mi
import numpy as np
from itertools import cycle
from more_itertools import take
import drjit as dr
from PIL import Image

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

def make_image_grid (images, row_major=True, gutter=True):
    """
    Make a large image where the images are stacked.

    images: list/list of list of PIL Images
    row_major: if images is list, whether to lay them down horizontally/vertically
    """
    assert isinstance(images, list) and len(images) > 0, "images is either not a list or an empty list"
    if isinstance(images[0], list) :
        return make_image_grid([make_image_grid(row, gutter=gutter) for row in images], False, gutter=gutter)
    else :
        if row_major :
            H = min(a.size[1] for a in images)
            images = [a.resize((int(H * a.size[0] / a.size[1]), H)) for a in images]
            W = sum(a.size[0] for a in images)
            gutter_width = int(0.01 * W) if gutter else 0
            W += (len(images) - 1) * gutter_width
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (cSum, 0))
                cSum += (a.size[0] + gutter_width)
        else :
            W = min(a.size[0] for a in images)
            images = [a.resize((W, int(W * a.size[1] / a.size[0]))) for a in images]
            H = sum(a.size[1] for a in images)
            gutter_width = int(0.01 * W) if gutter else 0
            H += (len(images) - 1) * gutter_width
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (0, cSum))
                cSum += (a.size[1] + gutter_width)
        return img

def fix_normal_and_tangent_map (n, t) : 
    """ 
    n and t are numpy arrays that are in the color mode. 
    [H, W, 3]

    Here we convert them into vectors and make sure that their
    norm is one and they are orthogonal
    """ 
    H, W, C = n.shape
    
    n = n / 255.0
    t = t / 255.0 

    n = (n * 2) - 1.0
    t = (t * 2) - 1.0

    n /= np.linalg.norm(n, axis=2, keepdims=True)

    t = t - (n * t).sum(axis=2, keepdims=True) * n # gram schmidt

    t /= np.linalg.norm(t, axis=2).reshape((H, W, 1))

    return n, t

def save_map_to_img (m, path='img.png')  :
    A = (m + 1) / 2
    Image.fromarray((A * 255.0).astype(np.uint8)).save(path)

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

def make_checker_board_texture (size, checker_size, colorA=(1, 0, 0,), colorB=(0, 0, 0)) :
    assert (size // checker_size) % 2 == 0, 'Checkerboard will not tile with given config' 
    img = np.zeros((size, size, 3))
    mask = take(size, cycle([False] * checker_size + [True] * checker_size))
    x_mask = np.array(mask)[None, :] 
    mask = x_mask ^ x_mask.T
    img[mask] = colorA
    img[~mask]= colorB
    return img

def rotate_s_mat (s_mat, normal, tangent) : 
    R = mi.Matrix3f(dr.cross(normal, tangent), tangent, normal)
    return dr.transpose(R) @ s_mat @ R

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
