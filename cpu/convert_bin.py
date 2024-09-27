import numpy as np
from PIL import Image


img = np.fromfile('out.bin', dtype=np.uint8)
img = img.reshape(1024, 1024, -1)
print(img.shape)
img = Image.fromarray(img)
img.save('out.png')