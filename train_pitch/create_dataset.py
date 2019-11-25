import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from group_three import *

train_three = open("train_three.csv", "w")
class_column = 0
for note in pitch:

    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        for row in thresh:
            for column in row:
                train_three.write(str(column) + ", ")

        train_three.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

train_three.close()

# ===============================================================

test_three = open("test_three.csv", "w")
class_column = 0
for note in test_pitch:

    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        for row in thresh:
            for column in row:
                test_three.write(str(column) + ", ")

        test_three.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

test_three.close()

exit()
