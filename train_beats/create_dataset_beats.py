import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import importlib

import helper
from group_beats import *

train_three = open("train_beats.csv", "w")
class_column = 0
for note in beats:

    class_counter = 0
    for i in note:
        print(i)
        img = cv2.imread(dataset_path + "..\\originals-resized\\" + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        # calculating histogram each col of image
        counts_col = np.sum(thresh == 255, axis=0)
        max_level = max(counts_col)
        min_level = min(counts_col)
        average_col = sum(counts_col)/30
        
        train_three.write(str(min_level) + ", ")
        train_three.write(str(max_level) + ", ")
        train_three.write(str(average_col) + ", ")

        train_three.write(str(class_column) + "\n")

        class_counter += 1
        if class_counter == 4:
            break

    class_column += 1

train_three.close()

# ===============================================================

test_three = open("test_beats.csv", "w")
class_column = 0
for note in test_beats:

    class_counter = 0
    for i in note:
        print(i)
        img = cv2.imread(dataset_path + "..\\originals-resized\\" + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # calculating histogram each col of image
        counts_col = np.sum(thresh == 255, axis=0)
        max_level = max(counts_col)
        min_level = min(counts_col)
        average_col = sum(counts_col)/30

        test_three.write(str(min_level) + ", ")
        test_three.write(str(max_level) + ", ")
        test_three.write(str(average_col) + ", ")

        test_three.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

test_three.close()

exit()
