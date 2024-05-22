import torch 
import math
from PIL import Image
import mitsuba as mi
import torch.optim as optim
import numpy as np 
import drjit as dr
from collections import defaultdict

VIS = True

def dot_torch (a, b) : 
    return (a * b).sum(dim=-1, keepdims=True)

def benny_rodrigues_rot_formula (v, k, theta) :
    """ this is benjamin rodrigues's formula for rotating vector v 
    around axis k (which is unit norm) by theta degrees """
    theta_ = theta.reshape(theta.shape + (1,))
    return v * torch.cos(theta_) \
        + torch.cross(k, v, dim=-1) * torch.sin(theta_) \
        + k * dot_torch(k, v) * (1 - torch.cos(theta_))

def benny_rodrigues_rot_formula_dr (v, k, theta) :
    """ this is benjamin rodrigues's formula for rotating vector v 
    around axis k (which is unit norm) by theta degrees for drjit """
    return v * dr.cos(theta) \
        + dr.cross(k, v) * dr.sin(theta) \
        + k * dr.dot(k, v) * (1 - dr.cos(theta))

def spherical_to_euclidean (v) : 
    """ convert spherical coordinates to euclidean """ 
    phi = v[..., 0]
    theta = v[..., 1]

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi) 

    x = x.reshape(phi.shape + (1,))
    y = y.reshape(phi.shape + (1,))
    z = z.reshape(phi.shape + (1,))

    return torch.cat((x, y, z), axis=-1)

def spherical_to_euclidean_dr (v) : 
    """ convert spherical coordinates to euclidean """ 
    phi = v[0]
    theta = v[1]

    x = dr.sin(phi) * dr.cos(theta)
    y = dr.sin(phi) * dr.sin(theta)
    z = dr.cos(phi) 

    return mi.Vector3f(x, y, z) 

def euclidean_to_spherical (xyz) : 
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    phi = torch.acos(z) 
    theta = torch.sign(y) * torch.acos(x / torch.sqrt(x ** 2 + y ** 2))

    phi = phi.reshape(phi.shape + (1,))
    theta = theta.reshape(theta.shape + (1,))

    return torch.cat((phi, theta), axis=-1)

def euclidean_to_spherical_dr (xyz) : 
    x = xyz.x
    y = xyz.y
    z = xyz.z

    phi = dr.acos(z) 
    theta = dr.mulsign(dr.acos(x / dr.safe_sqrt(x ** 2 + y ** 2)), y)

    return mi.Vector2f(phi, theta) 

def asg_torch (mu, gamma, log_sigma_x, log_sigma_y, C, d, return_ceiling=False) : 
    """
    mu - mean vector
        mu[..., 0] = polar angle - phi
        mu[..., 1] = azimuthal angle - theta
    gamma - rotation angle
    log_sigma_x - log of x bandwidth
    log_sigma_y - log of y bandwidth
    C - amplitude
    d - direction that we are querying
        d[..., 0] = polar angle
        d[..., 1] = azimuthal angle
    """
    mu_phi = mu[..., 0]
    mu_theta = mu[..., 1]

    # canonical x direction
    x_cano_phi = mu_phi + (torch.pi / 2)
    x_cano_theta = mu_theta
    x_cano = torch.stack((x_cano_phi, x_cano_theta), dim=-1)

    mu_vec = spherical_to_euclidean(mu)
    x_cano_vec = spherical_to_euclidean(x_cano)

    x_vec = benny_rodrigues_rot_formula(x_cano_vec, mu_vec, gamma)
    y_vec = torch.cross(mu_vec, x_vec, dim=-1)

    d_vec = spherical_to_euclidean(d)

    vals = C * torch.clamp(dot_torch(mu_vec, d_vec), min=0).squeeze() \
        * torch.exp(\
            -(torch.exp(log_sigma_x)) * (dot_torch(d_vec, x_vec) ** 2).squeeze() \
            -(torch.exp(log_sigma_y)) * (dot_torch(d_vec, y_vec) ** 2).squeeze())

    if return_ceiling: 
        max_vals = C * torch.exp(-torch.exp(log_sigma_x) - torch.exp(log_sigma_y))
        return vals, max_vals
        
    return vals

def asg_dr (mu, gamma, log_sigma_x, log_sigma_y, C, d) : 
    """
    mu - mean vector
        mu[0] = polar angle - phi
        mu[1] = azimuthal angle - theta
    gamma - rotation angle
    log_sigma_x - log of x bandwidth
    log_sigma_y - log of y bandwidth
    C - amplitude
    d - direction that we are querying
        d[0] = polar angle
        d[1] = azimuthal angle
    """
    mu_phi = mu.x
    mu_theta = mu.y

    # canonical x direction
    x_cano_phi = mu_phi + (dr.pi / 2)
    x_cano_theta = mu_theta
    x_cano = mi.Vector2f(x_cano_phi, x_cano_theta)

    mu_vec = spherical_to_euclidean_dr(mu)
    x_cano_vec = spherical_to_euclidean_dr(x_cano)

    x_vec = benny_rodrigues_rot_formula_dr(x_cano_vec, mu_vec, gamma)
    y_vec = dr.cross(mu_vec, x_vec)

    d_vec = spherical_to_euclidean_dr(d)

    vals = C * dr.clamp(dr.dot(mu_vec, d_vec), 0, 2) \
        * dr.exp(\
            -(dr.exp(log_sigma_x)) * (dr.dot(d_vec, x_vec) ** 2) \
            -(dr.exp(log_sigma_y)) * (dr.dot(d_vec, y_vec) ** 2))
    return vals

def test_spherical () : 
    vec = spherical_to_euclidean_dr(mi.Vector2f(0.0, 0.0))
    ans = mi.Vector3f(0, 0, 1)
    assert (dr.norm(ans - vec).numpy()[0] < 1e-7)

    vec = spherical_to_euclidean_dr(mi.Vector2f(np.pi / 2, 0.0))
    ans = mi.Vector3f(1, 0, 0)
    assert (dr.norm(ans - vec).numpy()[0] < 1e-7)

    vec = spherical_to_euclidean_dr(mi.Vector2f(np.pi / 2, np.pi))
    ans = mi.Vector3f(-1, 0, 0)
    assert (dr.norm(ans - vec).numpy()[0] < 1e-7)

    vec = spherical_to_euclidean_dr(mi.Vector2f(np.pi / 2, np.pi / 2))
    ans = mi.Vector3f(0, 1, 0)
    assert (dr.norm(ans - vec).numpy()[0] < 1e-7)

def test_sp_eu_conversion () : 
    rng = np.random.RandomState(0)

    for _ in range(10) : 
        phi = rng.uniform(0, np.pi)
        theta = rng.uniform(0, 2 * np.pi)

        vec = euclidean_to_spherical_dr(
            spherical_to_euclidean_dr(mi.Vector2f(phi, theta))
        )

        phi_ = vec.x.numpy()[0]
        theta_ = vec.y.numpy()[0]

        assert np.isclose(np.sin(phi), np.sin(phi_))
        assert np.isclose(np.cos(phi), np.cos(phi_))

        assert np.isclose(np.sin(theta), np.sin(theta_))
        assert np.isclose(np.cos(theta), np.cos(theta_))

def test_eu_sp_conversion () : 
    rng = np.random.RandomState(0)
    for _ in range(10) : 
        vec = mi.Vector3f(rng.randn(), rng.randn(), rng.randn())
        vec /= dr.norm(vec)
        vec_ = spherical_to_euclidean_dr(
            euclidean_to_spherical_dr(vec)
        )
        ans = dr.norm(vec - vec_).numpy()[0]
        assert np.isclose(ans, 0., atol=1e-5)

def test_rodrigues_formula () : 
    rng = np.random.RandomState(0) 
    for _ in range(10) :
        vec = mi.Vector3f(rng.randn(), rng.randn(), rng.randn())
        vec /= dr.norm(vec)

        sph = euclidean_to_spherical_dr(vec) 
        sph.x += dr.pi / 2

        perp_vec = spherical_to_euclidean_dr(sph)
        
        assert np.isclose(dr.dot(perp_vec, vec).numpy()[0], 0., atol=1e-5)
        
        theta = rng.uniform(0, 2 * np.pi)
        new_vec = benny_rodrigues_rot_formula_dr(perp_vec, vec, theta) 

        assert np.isclose(dr.dot(new_vec, vec).numpy()[0], 0., atol=1e-5)

        assert np.isclose(dr.dot(new_vec, perp_vec).numpy()[0], np.cos(theta), atol=1e-5)


if __name__ == "__main__": 
    mu = torch.rand(5, 5, 2)
    mu[..., 0] *= torch.pi
    mu[..., 1] *= torch.pi * 2.0

    gamma = torch.randn(5, 5) * torch.pi * 2.0

    log_sigma_x = torch.zeros_like(gamma)
    log_sigma_y = torch.zeros_like(gamma)

    C = torch.ones_like(gamma)

    d = torch.rand(5, 5, 2)
    d[..., 0] *= torch.pi
    d[..., 1] *= torch.pi * 2.0

    # print(asg_torch (mu, gamma, log_sigma_x, log_sigma_y, C, d))

    data = defaultdict(lambda : [])
    with open('data.txt') as fp : 
        for line in fp.readlines() : 
            y, x, phi, theta, V = line.split()
            y, x, phi, theta, V = int(y), int(x), float(phi), float(theta), float(V) 
            data[(y,x)].append((phi, theta, V))
    
    data_ten = np.ones((128, 128, 110, 3)) 
    for k, v in data.items(): 
        y, x = k  
        data_ten[x, y, ...] = np.array(v)

    if VIS : 
        cnt = 0
        for phi_i in range(11):
            for theta_i in range(10): 
                phi = phi_i * (math.pi / 2) * (1.0 / 10); 
                theta = theta_i * 2 * math.pi * (1.0 / 10); 
                Image.fromarray(data_ten[:, :, cnt, 2].astype(np.uint8) * 255, mode='L').save(f'vis/img_{phi:.2f}_{theta:.2f}.png')
                cnt += 1

    with torch.cuda.device('cuda') :
        data_ten = torch.from_numpy(data_ten)

        normals = torch.from_numpy((np.array(Image.open('bent_normal_map_128.png').convert('RGB')) / 255.) * 2 - 1)
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)
        # now simple optimization loop 
        mu_init = euclidean_to_spherical(normals)
        mu = torch.nn.Parameter(torch.clone(mu_init))
        gamma = torch.nn.Parameter(torch.zeros(128, 128))
        log_sigma_x = torch.nn.Parameter(torch.zeros(128, 128))
        log_sigma_y = torch.nn.Parameter(torch.zeros(128, 128))
        C = torch.ones(128, 128)

        params = [gamma, log_sigma_x, log_sigma_y]
        opt = optim.Adam(params, lr=1e-1) 

        for i in range(500) : 
            opt.zero_grad()

            indices = torch.randint(0, 110, (128, 128), dtype=torch.long)
            selected = data_ten[torch.arange(128).unsqueeze(1), torch.arange(128), indices]

            d = selected[..., :2]
            V = selected[..., 2].bool()

            pred, max_fn_val = asg_torch(mu, gamma, log_sigma_x, log_sigma_y, C, d, return_ceiling=True) 

            loss = -pred[V].sum() + pred[~V].sum()

            if i % 100 == 0 : 
                print(f'Iteration = {i}, Current loss = {loss.item():4f}')
                with torch.no_grad() : 
                    mu_ = mu.unsqueeze(2).repeat((1, 1, 110, 1))
                    gamma_ = gamma.unsqueeze(2).repeat(1, 1, 110)
                    log_sigma_x_ = log_sigma_x.unsqueeze(2).repeat(1, 1, 110)
                    log_sigma_y_ = log_sigma_y.unsqueeze(2).repeat(1, 1, 110)
                    C_ = C.unsqueeze(2).repeat(1, 1, 110)
                    preds, max_fn_vals = asg_torch(mu_, gamma_, log_sigma_x_, log_sigma_y_, C_, data_ten[..., :2], return_ceiling=True)
                    gt = data_ten[...,2].bool()
                    thresholds = [0.0, 0.1, 0.3, 0.5, 0.9]
                    print(f'Class bias - {gt.float().mean():.2f}')
                    print('Accuracy with absolutes:')  
                    print(' '.join([f'Thr - {_:.2f}, Acc - {((preds > _) == gt).float().mean().item():2f}' for _ in thresholds]))
                    print('Accuracy with Relatives:')  
                    print(' '.join([f'Thr - {_:.2f}, Acc - {(((preds / max_fn_vals) > _) == gt).float().mean().item():2f}' for _ in thresholds]))

            loss.backward() 
            opt.step()

        params = torch.stack((gamma, log_sigma_x, log_sigma_y), dim=-1).detach().cpu().numpy()
        #np.save('asg_params.npy', params)
        
        if VIS :
            with torch.no_grad() : 
                for phi_i in range(11):
                    for theta_i in range(10): 
                        phi = phi_i * (math.pi / 2) * (1.0 / 10); 
                        theta = theta_i * 2 * math.pi * (1.0 / 10); 
                        d = np.ones((128, 128, 2))
                        d[..., 0] = phi
                        d[..., 1] = theta
                        d = torch.from_numpy (d)
                        pred = asg_torch(mu, gamma, log_sigma_x, log_sigma_y, C, d).detach().cpu().numpy()
                        Image.fromarray((pred * 255).astype(np.uint8), mode='L').save(f'vis/pred_{phi:.2f}_{theta:.2f}.png')

