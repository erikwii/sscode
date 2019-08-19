import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from grouping import *

bimbelx = open("bimbelx.csv", "w")
class_column = 0
for note in c1:
    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
        for row in thresh:
            for column in row:
                bimbelx.write(str(column) + ", ")
        bimbelx.write(str(class_column) + "\n")
        class_counter += 1
        if class_counter == 20:
            break
    class_column += 1
bimbelx.close()

img_bank = list()
title_bank = list()

data_test = open("test.csv", 'w')
class_column = 0
for note in c1:
    class_counter = 0
    imgs = list()
    titles=list()
    for i in note:
        if (class_counter == 21) or (class_counter == 22):
            img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
            thresh_tmp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
            imgs.append(thresh_tmp)
            titles.append(str(class_column)+str(i))
            for row in thresh:
                for column in row:
                    data_test.write(str(column) + ", ")
            data_test.write(str(class_column) + "\n")
        class_counter += 1
        
        if class_counter == 23:
            break
    img_bank.append(imgs)
    title_bank.append(titles)
    class_column += 1
data_test.close()

for i in range(len(img_bank)):
    for j in range(len(img_bank[0])):
        plt.subplot(len(img_bank), len(img_bank[0]), (i*j)+1), plt.imshow(img_bank[i][j], 'gray')
        plt.title(title_bank[i][j])
        plt.xticks([]), plt.yticks([])
plt.show()
