from PIL import Image

png_path = "" + ".png"

im = Image.open(png_path, "r")
print(list(im.getdata())[:20])
