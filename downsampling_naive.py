"""
Naive implementation which takes a normal map of size 2^l
and generates normal maps of size 2^0 to 2^(l-1)
by simply averaging the 4 surrounding normals
"""



import numpy as np
from utils import read_txt_feature_map



def is_power_of_two(x):
    if x < 1:
        return False
    if x == 1:
        return True
    return is_power_of_two(x/2)



input_path = "cloth/normal_map.txt"
output_path = "mipmaps/"

normal_map = read_txt_feature_map(input_path)
normal_map_shape = normal_map.shape
if normal_map_shape[0] != normal_map_shape[1]:
    raise RuntimeError("Incorrect normal map shape: is should be square.")
normal_map_dim = normal_map_shape[0]
if not is_power_of_two(normal_map_dim):
    raise RuntimeError("Incorrect normal map dimension: it should be a power of 2.")

# NOTE: it could be helpful to save the input normal map as .npy as well
# np.save(output_path + "normal_" + str(normal_map_dim), normal_map)

# mipmap dims, starting from (input map dim)//2
cur_dim = normal_map_dim // 2
dims = []
while (cur_dim >= 1):
    dims.append(cur_dim)
    cur_dim //= 2

DIM_SAVE_VEC = 3
maps = [normal_map]
for dim in dims:
    cur_map = np.zeros((dim, dim, DIM_SAVE_VEC))
    prev_map = maps[-1]
    for i in np.arange(dim):
        for j in np.arange(dim):
            cur_map[i][j] = (
                prev_map[i*2][j*2] +
                prev_map[i*2+1][j*2] +
                prev_map[i*2][j*2+1] +
                prev_map[i*2+1][j*2+1]
            ) / 4.0
    maps.append(cur_map)
    np.save(output_path + "normal_" + str(dim), cur_map)
