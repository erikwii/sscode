import cv2
import numpy as np

# Load the image
img = cv2.imread('img/originals-resized/note-eighth-a1-167.png')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray, 1)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh, None, iterations=3)
thresh = cv2.erode(thresh, None, iterations=2)

# Find the contours
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.rectangle(thresh_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

scale_percent = 300 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Finally show the image
cv2.imshow('img', resized)
# cv2.imshow('res', thresh_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
