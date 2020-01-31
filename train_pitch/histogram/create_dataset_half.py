import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import importlib

import helper
from group_half import *

train_three = open("train_half.csv", "w")
class_column = 0
for note in pitch:

    class_counter = 0
    for i in note:
        # print(dataset_path + i)
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        # calculating histogram each row of image
        counts = np.sum(thresh == 255, axis=1)
        max_hist = max(counts)

        # calculate average of counts (head of notes position)
        average = sum(counts)/50
        check = (np.abs(counts - average) < 2)
        indices = [i for i, x in enumerate(check) if x == True]
        group_average = list(helper.split_tol(indices,2))
        average_index = sum(indices)/len(indices)

        # check the index row of the paranada exist
        paranada = (np.abs(counts - max_hist) <= 4)
        indices = [i for i, x in enumerate(paranada) if x == True]
        group_paranada = list(helper.split_tol(indices,2))

        # Check if there's dataset anomali
        if len(group_paranada) < 5:
            print(i)
            scale_percent = 500  # percent of original size
            width = int(thresh.shape[1] * scale_percent / 100)
            height = int(thresh.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('gray', resized)
            cv2.waitKey(0)
            y = range(49, -1, -1)
            plt.plot(counts, y)
            plt.show()
            
        paranada_index = [i[0] for i in group_paranada]

        # printable = {}
        # printable["average"] = average
        # printable["average_index"] = average_index
        # printable["paranada"] = paranada
        # printable["paranada_index"] = paranada_index
        # print(printable)
        # exit()

        # inserting feature to csv file
        for paranada in paranada_index:
            train_three.write(str(paranada) + ", ")
        
        train_three.write(str(average) + ", ")
        train_three.write(str(average_index) + ", ")

        train_three.write(str(class_column) + "\n")

        class_counter += 1
        if class_counter == 4:
            break

    class_column += 1

train_three.close()

# ===============================================================

test_three = open("test_half.csv", "w")
class_column = 0
for note in test_pitch:

    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # calculating histogram each row of image
        counts = np.sum(thresh == 255, axis=1)
        max_hist = max(counts)

        # calculate average of counts (head of notes position)
        average = sum(counts)/50
        check = (np.abs(counts - average) < 2)
        indices = [i for i, x in enumerate(check) if x == True]
        group_average = list(helper.split_tol(indices, 2))
        average_index = sum(indices)/len(indices)

        # check the index row of the paranada exist
        paranada = (np.abs(counts - max_hist) <= 4)
        indices = [i for i, x in enumerate(paranada) if x == True]
        group_paranada = list(helper.split_tol(indices, 2))
        paranada_index = [i[0] for i in group_paranada]

        # inserting feature to csv file
        for paranada in paranada_index:
            test_three.write(str(paranada) + ", ")

        test_three.write(str(average) + ", ")
        test_three.write(str(average_index) + ", ")

        test_three.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

test_three.close()

exit()
