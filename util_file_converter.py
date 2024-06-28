"""
Convert the file at `input_path`
to a file at `output_path`
The input/output file type must be in {txt, png, npy}

Currently Supports:
  txt to png
  txt to npy
  png to npy
  npy to png
"""

import argparse
import numpy as np
from utils import read_txt_feature_map
from PIL import Image

if __name__ == "__main__":
  parser = argparse.ArgumentParser("File Converter")
  parser.add_argument("input_path", type=str, help="Path of the input file.")
  parser.add_argument("output_path", type=str, help="Path of the output file.")

  args = parser.parse_args()
  input_path = args.input_path
  if (not (input_path.endswith(".txt") or input_path.endswith(".png") or input_path.endswith(".npy"))):
    raise RuntimeError("Input file must be txt, png, or npy")

  output_path = args.output_path
  if (not (output_path.endswith(".txt") or input_path.endswith(".png") or input_path.endswith(".npy"))):
    raise RuntimeError("Output file must be txt, png, or npy")

  # Cases
  if (input_path.endswith(".txt")):
    if (output_path.endswith(".png")):
      nm = read_txt_feature_map(input_path)
      im = Image.fromarray(np.uint8(nm * 255.0))
      im.save(output_path)
    elif (output_path.endswith(".npy")):
      nm = read_txt_feature_map(input_path)
      np.save(output_path, nm)
    else:
      raise NotImplementedError("Conversion currently not supported.")
  
  elif (input_path.endswith(".png")):
    if (output_path.endswith(".npy")):
      im = Image.open(input_path).convert('RGB')
      nm = np.array(im, dtype=float)
      nm /= 255.0
      np.save(output_path, nm)
    else:
      raise NotImplementedError("Conversion currently not supported.")
  
  elif (input_path.endswith(".npy")):
    if (output_path.endswith(".png")):
      nm = np.load(input_path)
      im = Image.fromarray(np.uint8(nm * 255.0))
      im.save(output_path)
    else:
      raise NotImplementedError("Conversion currently not supported.")
  
  else:
    raise NotImplementedError("Conversion currently not supported.")
