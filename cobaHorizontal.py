import cv2
import math

def scale_img(_img, percentage):
    scale_percent = percentage  # percent of original size
    width = int(_img.shape[1] * scale_percent / 100)
    height = int(_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(_img, dim, interpolation=cv2.INTER_AREA)
    return resized

img = cv2.imread('img/originals-resized/note-quarter-f2-1731.png',
                 cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 80, 120)
lines = cv2.HoughLinesP(edges, 1, math.pi/2, 80, None, 30, 1)
x5 = scale_img(edges,500)
cv2.imshow("edges", x5)
cv2.waitKey(0)
print(lines)
cv2.imwrite("1.png", img)
for line in lines:
    pt1 = (line[0][0],line[0][1])
    pt2 = (line[0][2],line[0][3])
    cv2.line(img, pt1, pt2, (0,0,255), 3)
cv2.imwrite("2.png", img)
