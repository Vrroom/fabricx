import numpy as np
from PIL import Image
from utils import read_txt_feature_map, tile_feature_map

input_path = "normal_map.txt"
output_path = "cloth/normal_map.png"

normal_map_unit = read_txt_feature_map(input_path)

times = 32
normal_map = tile_feature_map(normal_map_unit, times)
print(normal_map.shape)

im = Image.fromarray(np.uint8(normal_map * 255.0))
im.save(output_path)
