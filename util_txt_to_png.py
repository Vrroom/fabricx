import argparse
import numpy as np
from utils import read_txt_feature_map
from PIL import Image

if __name__ == "__main__":
  parser = argparse.ArgumentParser("TXT to PNG")
  parser.add_argument("input_path", type=str, help="Path of the input TXT file.")
  parser.add_argument("output_path", type=str, default="temp.png", help="Path of the output PNG file.")

  args = parser.parse_args()
  txt_path = args.input_path
  if (not txt_path.endswith(".txt")):
    raise RuntimeError("Input file name must end with .txt")

  png_path = args.output_path
  if (not png_path.endswith(".png")):
    raise RuntimeError("Output file name must end with .png")

  nm = read_txt_feature_map(txt_path)
  im = Image.fromarray(np.uint8(nm * 255.0))
  im.save(png_path)
