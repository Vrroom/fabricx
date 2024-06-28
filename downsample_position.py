"""
Downsampling from position maps
Consistent with the papers:
2019 Wu et al.
and
2023 Zhu et al.

TODO: currently not working as intended
"""

import numpy as np
from utils import read_txt_feature_map

def is_power_of_two(x):
    if x < 1:
        return False
    if x == 1:
        return True
    return is_power_of_two(x/2)

input_path = "cloth/position_map.txt"
output_path = "mipmaps/"

position_map_3d = read_txt_feature_map(input_path)
position_map_shape = position_map_3d.shape
if position_map_shape[0] != position_map_shape[1]:
    raise RuntimeError("Incorrect position map shape: it should be square.")
position_map_dim = position_map_shape[0]
if not is_power_of_two(position_map_dim-1):
    raise RuntimeError("Incorrect position map dimension: it should be (power of 2) + 1.")
position_map = np.sum(position_map_3d, axis=-1) # works since they are of format [0, 0, h], 0 + 0 + h = h

slope_map_dim = position_map_dim-1
slope_map = np.zeros((position_map_dim-1, position_map_dim-1, 2))
for i in np.arange(slope_map_dim):
    for j in np.arange(slope_map_dim):
        h00 = position_map[i][j]
        h01 = position_map[i][j+1]
        h10 = position_map[i+1][j]
        h11 = position_map[i+1][j+1]
        cur_slope = np.array([
            (h11 + h10 - h01 - h00) / 2.0,
            (h11 + h01 - h10 - h00) / 2.0
        ])
        slope_map[i][j] = cur_slope

slope_map *= slope_map_dim  # equivalent to /(1.0/slope_map_dim), to get rise over run

# TODO: currently not working as intended
# normal_map = np.zeros((position_map_dim-1, position_map_dim-1, 3))
# for i in np.arange(slope_map_dim):
#     for j in np.arange(slope_map_dim):
#         normal_map[i][j] = np.cross(
#             [1.0, 0.0, slope_map[i][j][0]],
#             [0.0, 1.0, slope_map[i][j][1]]
#         )

# normal_map /= np.linalg.norm(normal_map, axis=-1)[:,:,None]
# np.save(output_path + "normal_map_temp.npy", normal_map)
