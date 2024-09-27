from PIL import Image
import numpy as np

img = Image.open('image.jpg')
# crop and resize image to 1024x1024
img = img.crop((0, 0, 1024, 1024))
img.save('saved.jpg')
img = np.array(img)
print(img.shape)
img.tofile('image.bin')