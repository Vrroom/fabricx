"""
Naive implementation which takes a normal map of size 2^l
and generates normal maps of size 2^(l-1) to 2^0
by simply averaging the normals in the corresponding region
"""

import numpy as np
from utils import read_txt_feature_map
from PIL import Image

def is_power_of_two(x):
    if x < 1:
        return False
    if x == 1:
        return True
    return is_power_of_two(x/2)

output_path = "mipmaps/"

normal_map = (np.array(Image.open("normal_map.png").convert("RGB")) / 255.0) * 2.0 - 1.0
# normal_map = read_txt_feature_map("normal_map.txt") * 2.0 - 1.0
normal_map_shape = normal_map.shape
if normal_map_shape[0] != normal_map_shape[1]:
    raise RuntimeError("Incorrect normal map shape: it should be square.")
normal_map_dim = normal_map_shape[0]
if not is_power_of_two(normal_map_dim):
    raise RuntimeError("Incorrect normal map dimension: it should be a power of 2.")

# NOTE: it could be helpful to save the input normal map as .npy as well
# np.save(output_path + "normal_" + str(normal_map_dim), normal_map)

# mipmap dims, starting from (input map dim)//2
cur_dim = normal_map_dim // 2
cur_size = 2
dims = []
sizes = []
while (cur_dim >= 1):
    dims.append(cur_dim)
    sizes.append(cur_size)
    cur_dim //= 2
    cur_size *= 2

DIM_SAVE_VEC = 3
# maps = [normal_map]
for dim, size in zip(dims, sizes):
    cur_map = np.zeros((dim, dim, DIM_SAVE_VEC))
    for i1 in np.arange(dim):
        for j1 in np.arange(dim):
            cur_sum = np.zeros(DIM_SAVE_VEC)
            for i2 in np.arange(i1*size, (i1+1)*size):
                for j2 in np.arange(j1*size, (j1+1)*size):
                    cur_sum += normal_map[i2][j2]
            cur_avg = cur_sum / float(size**2)
            cur_map[i1][j1] = cur_avg / np.linalg.norm(cur_avg)

    # Previous Implementation:
    # prev_map = maps[-1]
    # for i in np.arange(dim):
    #     for j in np.arange(dim):
    #         cur_map[i][j] = (
    #             prev_map[i*2][j*2] +
    #             prev_map[i*2+1][j*2] +
    #             prev_map[i*2][j*2+1] +
    #             prev_map[i*2+1][j*2+1]
    #         ) / 4.0
    # maps.append(cur_map)
    # cur_map_normalized = cur_map / np.linalg.norm(cur_map, axis=-1)[:,:,None]   # only normalize when storing, to avoid incorrect averaging
    # np.save(output_path + "normal_" + str(dim) + ".npy", cur_map)
    im = Image.fromarray(np.uint8(((cur_map + 1.0) / 2.0) * 255.0))
    im.save(output_path + "normal_" + str(dim) + ".png")
