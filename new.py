import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from group_beat import *
from group_pitch import *

randi = [
    random.sample(range(0,49), 40),     # whole
    random.sample(range(0,449), 40),    # half
    random.sample(range(0,1184), 40),   # quarter
    random.sample(range(0,1304), 40),   # eighth
    random.sample(range(0,276), 40)     # sixteenth
]

train_beats = open("train_beats.csv", "w")
class_column = 0
for note in beats:

    class_counter = 0
    for i in note:
        if class_counter in randi[class_column]:
            img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
            for row in thresh:
                for column in row:
                    train_beats.write(str(column) + ", ")
            
            train_beats.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

train_beats.close()

# ===============================================================

randi = [
    random.sample(range(0, 173), 20),  # c1
    random.sample(range(0, 255), 20),  # d1
    random.sample(range(0, 398), 20),  # e1
    random.sample(range(0, 366), 20),  # f1
    random.sample(range(0, 371), 20),  # g1
    random.sample(range(0, 427), 20),  # a1
    random.sample(range(0, 366), 20),  # b1
    random.sample(range(0, 268), 20),  # c2
    random.sample(range(0, 243), 20),  # d2
    random.sample(range(0, 145), 20),  # e2
    random.sample(range(0, 54), 20),   # f2
    random.sample(range(0, 22), 20),   # g2
    random.sample(range(0, 20), 20),   # a2
    random.sample(range(0, 10), 10),   # b2
    random.sample(range(0, 1), 1)      # c3
]

train_pitch = open("train_pitch.csv", "w")
class_column = 0
for note in pitch:

    class_counter = 0
    for i in note:
        if class_counter in randi[class_column]:
            img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            for row in thresh:
                for column in row:
                    train_pitch.write(str(column) + ", ")

            train_pitch.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

train_pitch.close()

exit()

img_bank = list()
title_bank = list()

data_test = open("test.csv", 'w')
class_column = 0
for note in beats:
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

# for i in range(len(img_bank)):
#     for j in range(len(img_bank[0])):
#         plt.subplot(len(img_bank), len(img_bank[0]), (i*j)+1), plt.imshow(img_bank[i][j], 'gray')
#         plt.title(title_bank[i][j])
#         plt.xticks([]), plt.yticks([])
# plt.show()
