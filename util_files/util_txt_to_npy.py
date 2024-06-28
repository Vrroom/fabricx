"""
Convert the txt file at ../`input_path`
to a npy file at ../`output_path`
i.e. please enter the paths relative to the main directory
"""

import argparse
import numpy as np
from utils import read_txt_feature_map

if __name__ == "__main__":
  parser = argparse.ArgumentParser("TXT to NPY")
  parser.add_argument("input_path", type=str, help="Path of the input TXT file.")
  parser.add_argument("output_path", type=str, default="temp.npy", help="Path of the output NPY file.")

  args = parser.parse_args()
  txt_path = "../" + args.input_path
  if (not txt_path.endswith(".txt")):
    raise RuntimeError("Input file name must end with .txt")

  npy_path = "../" + args.output_path
  if (not npy_path.endswith(".npy")):
    raise RuntimeError("Output file name must end with .npy")
  
  nm = read_txt_feature_map(txt_path)
  np.save(npy_path)
