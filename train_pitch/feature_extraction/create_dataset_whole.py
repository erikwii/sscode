import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import importlib

import helper
from group_whole import *

train_whole = open("train_whole.csv", "w")
class_column = 0
for note in pitch:

    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        # calculating histogram each row of image
        counts = np.sum(thresh == 255, axis=1)
        max_hist = max(counts)
        # helper.show_plot(i,counts, "")
        print(counts)
        non_paranada = list()
        # check the index row of the paranada exist
        mean = np.mean(counts)
        std = np.std(counts)
        stat_z = [(s-mean)/std for s in counts]
        paranada = (np.abs(stat_z) > 2)
        indices = [i for i, x in enumerate(paranada) if x == True]
        j = 0
        print(paranada)
        for x in paranada:
            if x == True:
                non_paranada.append(min(counts))
            else :
                non_paranada.append(counts[j])
            j += 1
        group_paranada = list(helper.split_tol(indices,2))
        paranada_index = [i[0] for i in group_paranada]
        print("paranada: ")
        print(paranada_index)

        # calculate average of counts (head of notes position)
        average = sum(counts)/50
        tolerance = 2
        mean = np.mean(non_paranada)
        std = np.std(non_paranada)
        stat_z = [(s - mean)/std for s in non_paranada]
        check = (np.abs(counts - average) <= tolerance)
        check = np.abs(stat_z) > 2
        # print(check)
        indices = [i for i, x in enumerate(check) if x == True]
        print(i)
        print(average)
        print(non_paranada)
        print(indices)
        # if indices == [] or tolerance > 3:
        group_average = list(helper.split_tol(indices,2))
        average_index = sum(indices)/len(indices)
        print("group_average: " + str(group_average))
        print("average_index: " + str(average_index))
        helper.show_non_paranada_plot(i, counts, non_paranada)
        if class_counter == 2:
            exit()
        # printable = {}
        # printable["average"] = average
        # printable["average_index"] = average_index
        # printable["paranada"] = paranada
        # printable["paranada_index"] = paranada_index
        # print(printable)
        # exit()

        # inserting feature to csv file
        for paranada in paranada_index:
            train_whole.write(str(paranada) + ", ")
        
        train_whole.write(str(average) + ", ")
        train_whole.write(str(average_index) + ", ")

        train_whole.write(str(class_column) + "\n")

        class_counter += 1
        if class_counter == 4:
            break

    class_column += 1

train_whole.close()
exit()
# ===============================================================

test_whole = open("test_whole.csv", "w")
class_column = 0
for note in test_pitch:

    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # calculating histogram each row of image
        counts = np.sum(thresh == 255, axis=1)

        # calculate average of counts (head of notes position)
        average = sum(counts)/50
        check = (np.abs(counts - average) < 2)
        indices = [i for i, x in enumerate(check) if x == True]
        group_average = list(helper.split_tol(indices, 2))
        average_index = sum(indices)/len(indices)

        # check the index row of the paranada exist
        paranada = (np.abs(counts - 30) <= 5)
        indices = [i for i, x in enumerate(paranada) if x == True]
        group_paranada = list(helper.split_tol(indices, 2))
        paranada_index = [i[0] for i in group_paranada]

        # inserting feature to csv file
        for paranada in paranada_index:
            test_whole.write(str(paranada) + ", ")

        test_whole.write(str(average) + ", ")
        test_whole.write(str(average_index) + ", ")

        test_whole.write(str(class_column) + "\n")

        class_counter += 1

    class_column += 1

test_whole.close()

exit()
