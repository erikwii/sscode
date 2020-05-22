import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading
import math
import glob
import os

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
    for i in range(0, 5):
        for j in average_list:
            for n in j:
                paranada_np = np.asarray(paranada_list[i])
                paranada = (np.abs(paranada_np - n) <= 2)
                count = paranada.tolist().count(True)
                if count > max:
                    max = count
                    index = i+1
    return index

def min_max_normalize(data, min=0, max=100):
    return round((data - min) / (max - min), 3)

def detect_paranada(hist, filename):
    mean = np.mean(hist)
    std = np.std(hist)
    stat_z = [(s-mean)/std for s in hist]
    
    paranada = (np.abs(stat_z) > 2)
    indices = [i for i, x in enumerate(paranada) if x == True]
    
    if indices == []:
        max_hist = max(hist)

        paranada = (np.abs(np.abs(hist - max_hist) <= 2))
        indices = [i for i, x in enumerate(paranada) if x == True]

        log_message = "WARNING: Failed to get outlier of " + filename
        print(log_message)
        # helper.show_plot(i, hist, "")
    
    group_paranada = list(split_tol(indices,2))
    paranada_index = [i[0] for i in group_paranada]

    if len(paranada_index) < 5:
        print("Something error with process " + filename)
    
    return paranada, group_paranada, paranada_index

def remove_paranada(paranada, hist):
    non_paranada = list()
    j = 0
    for x in paranada:
        if x == True:
            if j > 0 and j < len(paranada)-1:
                if paranada[j+1] == True:
                    mean = (hist[j-1]+hist[j+2])/2
                elif paranada[j-1] == True:
                    mean = (hist[j-2]+hist[j+1])/2
                else:
                    mean = (hist[j-1]+hist[j+1])/2
            elif j == 0:
                if paranada[j+1] == True:
                    mean = hist[j+2]
                else:
                    mean = hist[j+1]
            elif j == len(paranada)-1:
                if paranada[j-1] == True:
                    mean = hist[j-2]
                else:
                    mean = hist[j-1]
            non_paranada.append(mean)
        else :
            non_paranada.append(hist[j])
        j += 1

    return non_paranada

kernel = np.ones((5, 5), np.uint8)

# Import image from img folder
img_name = 'note-whole-f1-946'
img = cv2.imread('img/originals-resized/' + img_name + ".png",
                 cv2.IMREAD_GRAYSCALE)
thresh_normal = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
# f = open(img_name+".txt", "w")
# for row in img:
#     s = ""
#     for d in row:
#         normal = min_max_normalize(d, 0, 255)
#         if d != 255 and d != 0:
#             s += (str(normal) + "\t")
#         else:
#             s += (str(normal) + "\t\t")
#     f.write(s+"\n")
# f.close()
otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# calculating histogram each row of image
counts = np.sum(thresh == 255, axis=1)

mean = np.mean(counts)
# calculate average of counts (head of notes position)
check = (np.abs(counts - mean) < 2)
indices = [i for i, x in enumerate(check) if x == True]
group_average = list(split_tol(indices,2))
average_index = sum(indices)/len(indices)

scale_percent = 500  # percent of original size
width = int(thresh.shape[1] * scale_percent / 100)
height = int(thresh.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite(img_name + " (binary inverted).png", resized)

hd_hist = []
for i in range(50):
    if i not in indices:
        hd_hist.append(0)
    else:
        hd_hist.append(1)

paranada, group_paranada, index_paranada = detect_paranada(counts, img_name)
non_paranada = remove_paranada(paranada, counts)

y = range(49, -1, -1)
yticks = range(49, -1, -2)
plt.subplot(1, 2, 1)
plt.plot(counts, y)
plt.yticks(yticks)
plt.axis((0,30,0,50))
plt.title(img_name + ' (Asli)')

plt.subplot(1, 2, 2)
plt.plot(non_paranada, y)
plt.yticks(yticks)
plt.axis((0,30,0,50))
plt.title(img_name + ' (Non-paranada)')

plt.show()
exit()

max_hist = max(counts)
counts_col = np.sum(thresh == 255, axis=0)
print("Max level col: ", end="")
print(str(max(counts_col)))
print("Min level col: ", end="")
print(str(min(counts_col)))
average_col = sum(counts_col)/30
print("Avg level col: ", end="")
print(average_col)

# check the index row of the paranada exist
paranada = (np.abs(counts - max_hist) <= 2)
indices = [i for i, x in enumerate(paranada) if x == True]
print("\nParanada: ", end="")
print(indices)
group_paranada = list(split_tol(indices, 2))
print("Group of Paranada: ", end="")
print(group_paranada)

# calculate where is the head of notes exist
average = sum(counts)/50
print("\nAvrg hist value: " + str(average))
check = (np.abs(counts - average) < 2)
indices = [i for i, x in enumerate(check) if x == True]
print("Average point: ", end="")
print(indices)
group_average = list(split_tol(indices, 2))
print("Group of avrg point: ", end="")
print(group_average)

print("\nin line: " + str(find_paranada_index(group_paranada, group_average)))

# Print plot histogram
# exit()
# plt.plot(counts,y)
# plt.show()
# exit()
# plt.hist(thresh.ravel(),256,[0,256]); plt.show()

# resize image
resized_thresh = cv2.resize(thresh_normal, dim, interpolation=cv2.INTER_AREA)
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
resized_otsu = cv2.resize(otsu, dim, interpolation=cv2.INTER_AREA)

def show_plot():
    y = range(49, -1, -1)
    plt.subplot(1, 2, 1)
    plt.plot(counts, y)
    plt.title(img_name + ' (Row)')

    plt.subplot(1, 2, 2)
    x = range(30)
    plt.plot(x, counts_col)
    plt.title(img_name + ' (Col)')

    plt.show()


def show_plot_col():
    aplt = plt
    x = range(30)
    aplt.plot(x, counts_col)
    aplt.show()

numpy_horizontal = np.hstack((resized_img, resized_thresh))
numpy_horizontal = np.hstack((numpy_horizontal, resized))

thread1 = threading.Thread(target=show_plot)
thread1.start()

# cv2.imwrite(img_name + " (binary inverted)", resized)

cv2.imshow(img_name, numpy_horizontal)
cv2.imshow('Citra Otsu', resized_otsu)

plt.title(img_name); plt.hist(img.ravel(),256,[0,256]); plt.show()
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

scale_percent = 500  # percent of original size
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
