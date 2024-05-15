import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the tangent map from the PPM file
tangent_image = Image.open('tangent_map.ppm')
tangent_map = np.array(tangent_image) / 255.0 * 2 - 1  # Convert to [-1, 1] range

# Extract the x and y components of the tangent
U = tangent_map[:, :, 0]
V = tangent_map[:, :, 1]

# Generate a grid of coordinates
height, width = tangent_map.shape[:2]
Y, X = np.mgrid[0:height, 0:width]

density = 64

X = X[::density, ::density]
Y = Y[::density, ::density]
U = U[::density, ::density]
V = V[::density, ::density]

# Plot the image
plt.imshow(tangent_image, extent=(0, width, height, 0))

# Overlay the vector field
plt.quiver(X, Y, U, V, color='red', scale=50)

plt.show()

