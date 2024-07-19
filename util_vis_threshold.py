import numpy as np
from PIL import Image

v_threshold = 0.15
prediction = np.array(Image.open("prediction.png")) / 255.0
prediction_threshholded = np.where(prediction < v_threshold, 0.0, 1.0)
im = Image.fromarray(np.uint8(prediction_threshholded * 255.0))
im.save("thresholded.png")
