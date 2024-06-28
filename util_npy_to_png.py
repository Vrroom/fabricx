"""
Convert the npy file at `input_path`
to a png file at `output_path`
"""

import argparse
import numpy as np
from PIL import Image

if __name__ == "__main__":
  parser = argparse.ArgumentParser("NPY to PNG")
  parser.add_argument("input_path", type=str, help="Path of the input NPY file.")
  parser.add_argument("output_path", type=str, default="temp.png", help="Path of the output PNG file.")

  args = parser.parse_args()
  npy_path = args.input_path
  if (not npy_path.endswith(".npy")):
    raise RuntimeError("Input file name must end with .npy")

  png_path = args.output_path
  if (not png_path.endswith(".png")):
    raise RuntimeError("Output file name must end with .png")

  nm = np.load(npy_path)
  im = Image.fromarray(np.uint8(nm * 255.0))
  im.save(png_path)
