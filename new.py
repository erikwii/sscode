import cv2
import numpy as np
from matplotlib import pyplot as plt

path_to_img = 'C:/Users/X230/Desktop/sscode/originals-resized/'
img = cv2.imread(path_to_img + 'note-eighth-a1-998.png', 0)
edges = cv2.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
