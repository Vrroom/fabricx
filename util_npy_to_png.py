"""
for visualizing mipmaps
"""

import os
import numpy as np
from PIL import Image

mipmaps_path = "mipmaps/"

files = os.scandir(mipmaps_path)
for file in files:
    if file.name.endswith(".npy"):
        im = Image.fromarray(np.uint8(np.load(file.path) * 255.0))
        im.save(file.path.removesuffix(".npy") + ".png")

files.close()
