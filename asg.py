import torch 
import numpy as np 
import drjit as dr

def dot_torch (a, b) : 
    return (a * b).sum(dim=-1, keepdims=True)

def benny_rodrigues_rot_formula (v, k, theta) :
    """ this is benjamin rodrigues's formula for rotating vector v 
    around axis k (which is unit norm) by theta degrees """
    theta_ = theta.reshape(theta.shape + (1,))
    return v * torch.cos(theta_) \
        + torch.cross(k, v, dim=-1) * torch.sin(theta_) \
        + k * dot_torch(k, v) * (1 - torch.cos(theta_))

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

def asg_torch (mu, gamma, sigma_x, sigma_y, C, d) : 
    """
    mu - mean vector
        mu[..., 0] = polar angle - phi
        mu[..., 1] = azimuthal angle - theta
    gamma - rotation angle
    sigma_x - x bandwidth
    sigma_y - y bandwidth
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
            -sigma_x * (dot_torch(d_vec, x_vec) ** 2).squeeze() \
            -sigma_y * (dot_torch(d_vec, y_vec) ** 2).squeeze())
    return vals

if __name__ == "__main__": 
    mu = torch.rand(5, 5, 2)
    mu[..., 0] *= torch.pi
    mu[..., 1] *= torch.pi * 2.0

    gamma = torch.randn(5, 5) * torch.pi * 2.0

    sigma_x = torch.ones_like(gamma)
    sigma_y = torch.ones_like(gamma)

    C = torch.ones_like(gamma)

    d = torch.rand(5, 5, 2)
    d[..., 0] *= torch.pi
    d[..., 1] *= torch.pi * 2.0

    print(asg_torch (mu, gamma, sigma_x, sigma_y, C, d))
    
