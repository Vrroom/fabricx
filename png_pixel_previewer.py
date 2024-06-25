import argparse
from PIL import Image

if __name__ == "__main__":
  parser = argparse.ArgumentParser("PNG Previewer")
  parser.add_argument("path", type=str, help="Path of the PNG file.")
  parser.add_argument("count", type=int, help="Number of pixels printed.")

  args = parser.parse_args()
  png_path = args.path

  im = Image.open(png_path, "r")
  print(list(im.getdata())[:args.count])
