import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
print(next(os.walk(path))[1])
exit()
img = cv.imread('img/originals-resized/note-eighth-a1-167.png', cv.IMREAD_GRAYSCALE)
# img = cv.medianBlur(img, 5)
ret, th1 = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY_INV, 11, 2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY_INV, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
