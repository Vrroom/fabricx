import bpy 
import os
import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
from drawTools import *
import numpy as np 
from PIL import Image

bpy.ops.mesh.primitive_uv_sphere_add(radius=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].levels = 3
bpy.context.object.modifiers["Subdivision"].render_levels = 3
bpy.ops.object.modifier_apply(modifier="Subdivision")

color_map = np.array(Image.open('color_map.jpg'))
W = color_map.shape[1]

def benny_rodrigues_rot_formula(v, k, theta) :
    """ this is benjamin rodrigues's formula for rotating vector v 
    around axis k (which is unit norm) by theta degrees for drjit """
    return v * np.cos(theta) \
        + np.cross(k, v) * np.sin(theta) \
        + k * np.dot(k, v) * (1 - np.cos(theta))
        
def spherical_to_euclidean (v) : 
    """ convert spherical coordinates to euclidean """ 
    phi = v[0]
    theta = v[1]

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi) 

    return np.array((x, y, z))

def asg_distr (d_vec, mu, gamma, log_sigma_x, log_sigma_y, C) : 
    mu_phi = mu[0]
    mu_theta = mu[1]

    # canonical x direction
    x_cano_phi = mu_phi + (np.pi / 2)
    x_cano_theta = mu_theta
    x_cano = np.array((x_cano_phi, x_cano_theta))

    mu_vec = spherical_to_euclidean(mu)
    x_cano_vec = spherical_to_euclidean(x_cano)

    x_vec = benny_rodrigues_rot_formula(x_cano_vec, mu_vec, gamma)
    y_vec = np.cross(mu_vec, x_vec)

    vals = C * np.clip(np.dot(mu_vec, d_vec), 0, np.inf) \
        * np.exp(\
            -(np.exp(log_sigma_x)) * (np.dot(d_vec, x_vec) ** 2) \
            -(np.exp(log_sigma_y)) * (np.dot(d_vec, y_vec) ** 2))
    return vals

def sggx_pdf(wm, s_mat) : 
    det_s = np.abs(np.linalg.det(s_mat))
    den = wm.x * wm.x * (s_mat[1, 1] * s_mat[2, 2] - s_mat[1, 2] * s_mat[1, 2]) + \
          wm.y * wm.y * (s_mat[0, 0] * s_mat[2, 2] - s_mat[0, 2] * s_mat[0, 2]) + \
          wm.z * wm.z * (s_mat[0, 0] * s_mat[1, 1] - s_mat[0, 1] * s_mat[0, 1]) + \
          2. * (wm.x * wm.y * (s_mat[0, 2] * s_mat[1, 2] - s_mat[2, 2] * s_mat[0, 1]) + \
                wm.x * wm.z * (s_mat[0, 1] * s_mat[1, 2] - s_mat[1, 1] * s_mat[0, 2]) + \
                wm.y * wm.z * (s_mat[0, 1] * s_mat[0, 2] - s_mat[0, 0] * s_mat[1, 2]))
    return np.maximum(det_s, 0.) * np.sqrt(det_s) \
           / (np.pi * np.square(den))


def solid_texture_objects(objs, texture_fn=lambda *args, **kwargs : (1.0, 0.0, 0.0, 1.0), 
    as_emission=True, smooth=False, distribution='sggx'):
        
    m, M = get_obj_bounds_list(objs)
    m = [m.x, m.y, m.z]
    M = [M.x, M.y, M.z]
    
    if distribution == 'sggx': 
        s_mat = np.array([
            [1, 0, 0],
            [0, 0.81, 0],
            [0, 0, 1]
        ])
    else :
        mu = np.array([0, 0])
        gamma = np.array(-np.pi/4)
        log_sigma_x = np.array(np.log(300))
        log_sigma_y = np.array(np.log(3))
        C = np.array(1)
    
    
    scale_factor = max([a - b for a, b in zip(M, m)])
    for obj in objs :
        mesh = obj.data
        if not mesh.vertex_colors:
            mesh.vertex_colors.new()
        color_layer = mesh.vertex_colors.active
        pdfs = []
        for poly in mesh.polygons:
            for idx in poly.loop_indices:
                vertex_index = mesh.loops[idx].vertex_index
                vertex = mesh.vertices[vertex_index]
                px = obj.matrix_world @ vertex.co
                
                if distribution == 'sggx' : 
                    pdf = sggx_pdf(px, s_mat)
                else :
                    pdf = asg_distr(np.array((px.x, px.y, px.z)), mu, gamma, log_sigma_x, log_sigma_y, C)
                    
                pdfs.append(pdf)
                
        m = np.min(pdfs)
        M = np.max(pdfs)
        for poly in mesh.polygons:
            for idx in poly.loop_indices:
                vertex_index = mesh.loops[idx].vertex_index
                vertex = mesh.vertices[vertex_index]
                px = obj.matrix_world @ vertex.co
                
                if distribution == 'sggx' : 
                    val = sggx_pdf(px, s_mat)
                else :
                    val = asg_distr(np.array((px.x, px.y, px.z)), mu, gamma, log_sigma_x, log_sigma_y, C)
                    
                pdf = (val - m) / (M - m)
                cmap_id = min(int(pdf * W), W - 1)
                color = color_map[0, cmap_id] / 255.
                color_layer.data[idx].color = [color[0], color[1], color[2], 1.0]
        material_name = f'{obj.name}_mat'
        add_vertex_color_as_material(obj, material_name, as_emission)
        if smooth :
            shade_smooth(obj)
            
solid_texture_objects([bpy.context.object], distribution='asg')
