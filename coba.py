import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading

# helper generator
def split_tol(test_list, tol):
    res = []
    last = test_list[0]
    for ele in test_list:
        if ele-last > tol:
            yield res
            res = []
        res.append(ele)
        last = ele
    yield res

def find_paranada_index(paranada_list, average_list):
    max = 0
    index = -1
    for i in range(0,5):
        for j in average_list:
            for n in j:
                paranada_np = np.asarray(paranada_list[i])
                paranada = (np.abs(paranada_np - n) <= 2)
                count = paranada.tolist().count(True)
                if count > max:
                    max = count
                    index = i+1
    return index

kernel = np.ones((5, 5), np.uint8)

# Import image from img folder
img = cv2.imread('img/originals-resized/note-quarter-e2-23.png',
                 cv2.IMREAD_GRAYSCALE)
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# calculating histogram each row of image
counts = np.sum(thresh == 255, axis=1)

# check the index row of the paranada exist
paranada = (np.abs(counts - 30) <= 5)
indices = [i for i, x in enumerate(paranada) if x == True]
print(indices)
group_paranada = list(split_tol(indices,2))
print(group_paranada)

# calculate where is the head of notes exist
average = sum(counts)/50
print(average)
check = (np.abs(counts - average) < 1)
indices = [i for i, x in enumerate(check) if x == True]
print(indices)
group_average = list(split_tol(indices, 2))
print(group_average)

print("is: " + str(find_paranada_index(group_paranada, group_average)))

# Print plot histogram
# exit()
y = range(49,-1,-1)
# plt.plot(counts,y)
# plt.show()
# exit()
# plt.hist(thresh.ravel(),256,[0,256]); plt.show()

scale_percent = 500  # percent of original size
width = int(thresh.shape[1] * scale_percent / 100)
height = int(thresh.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

def show_plot():
    plt.plot(counts, y)
    plt.show()

thread1 = threading.Thread(target=show_plot)
thread1.start()

cv2.imshow('gray', resized)
cv2.waitKey(0)

exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
linek = np.zeros((11, 11), dtype=np.uint8)
linek[5, ...] = 1
x = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek, iterations=1)
gray -= x

# # Load the image
# img = cv2.imread('img/originals-resized/note-eighth-a1-167.png')

# # convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # smooth the image to avoid noises
# gray = cv2.medianBlur(gray, 1)

# # Apply adaptive threshold
# thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
# thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# # apply some dilation and erosion to join the gaps
# thresh = cv2.dilate(thresh, None, iterations=3)
# thresh = cv2.erode(thresh, None, iterations=2)

# # Find the contours
# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # For each contour, find the bounding rectangle and draw it
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
#     cv2.rectangle(thresh_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

scale_percent = 500 # percent of original size
width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('gray', resized)
cv2.waitKey(0)
# # Finally show the image
# cv2.imshow('img', resized)
# # cv2.imshow('res', thresh_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
