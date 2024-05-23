import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def color_to_vec(x):
    x = np.array(x) / 255.
    x = x[:-1]
    return x * 2 - 1

image_data = np.array(Image.open('normal_map.png'))

fig, ax = plt.subplots()
ax.imshow(image_data)

def on_hover(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < image_data.shape[1] and 0 <= y < image_data.shape[0]:
            pixel = image_data[y, x]
            transformed_pixel = color_to_vec(pixel)
            ax.set_title(f"Transformed pixel: {transformed_pixel}")
            fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_hover)
plt.show()

