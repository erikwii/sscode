import random
import math
import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import importlib
import helper

def group_data(identifier):
    path = os.path.dirname(os.path.abspath(__file__))

    dataset_path = path + "\\..\\..\\img\\"+ identifier +"\\"

    e1 = glob.glob1(dataset_path+"e1\\", "note-" + identifier + "*")
    f1 = glob.glob1(dataset_path+"f1\\", "note-" + identifier + "*")
    g1 = glob.glob1(dataset_path+"g1\\", "note-" + identifier + "*")
    a1 = glob.glob1(dataset_path+"a1\\", "note-" + identifier + "*")
    b1 = glob.glob1(dataset_path+"h1\\", "note-" + identifier + "*")
    c2 = glob.glob1(dataset_path+"c2\\", "note-" + identifier + "*")
    d2 = glob.glob1(dataset_path+"d2\\", "note-" + identifier + "*")
    e2 = glob.glob1(dataset_path+"e2\\", "note-" + identifier + "*")
    f2 = glob.glob1(dataset_path+"f2\\", "note-" + identifier + "*")

    train = [e1, f1, g1, a1, b1, c2, d2, e2, f2]

    test_e1 = glob.glob1(dataset_path+"e1\\test\\", "note-" + identifier + "*")
    test_f1 = glob.glob1(dataset_path+"f1\\test\\", "note-" + identifier + "*")
    test_g1 = glob.glob1(dataset_path+"g1\\test\\", "note-" + identifier + "*")
    test_a1 = glob.glob1(dataset_path+"a1\\test\\", "note-" + identifier + "*")
    test_b1 = glob.glob1(dataset_path+"h1\\test\\", "note-" + identifier + "*")
    test_c2 = glob.glob1(dataset_path+"c2\\test\\", "note-" + identifier + "*")
    test_d2 = glob.glob1(dataset_path+"d2\\test\\", "note-" + identifier + "*")
    test_e2 = glob.glob1(dataset_path+"e2\\test\\", "note-" + identifier + "*")
    test_f2 = glob.glob1(dataset_path+"f2\\test\\", "note-" + identifier + "*")

    test = [test_e1, test_f1, test_g1, test_a1, test_b1, test_c2, test_d2, test_e2, test_f2]

    return train, test, dataset_path

def create_csv(**kwargs):
    identifier = kwargs.get('identifier', "quarter")

    train_group, test_group, dataset_path = group_data(identifier)

    type = kwargs.get('type', 'all')

    if type == 'all' or type == 'train':
        train_data = open("train_"+ identifier +".csv", "w")
        train_info = open("train_"+ identifier +"_info.csv", "w")
        class_column = 0
        for note in train_group:

            class_counter = 0
            for i in note:
                img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)

                thresh_method = kwargs.get('thresh_method', "gaussian")
                if thresh_method == 'mean':
                    thresh_cv = cv2.ADAPTIVE_THRESH_MEAN_C
                else:
                    thresh_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

                thresh = cv2.adaptiveThreshold(img, 255, thresh_cv,
                                                cv2.THRESH_BINARY_INV, 11, 2)
                
                # calculating histogram each row of image
                hist = np.sum(thresh == 255, axis=1)
                
                # check the index row of the paranada exist
                paranada, group_paranada, paranada_index = detect_paranada(hist, i)

                # remove paranada
                non_paranada = remove_paranada(paranada, hist)
                non_paranada = remove_outlier(non_paranada)

                # calculate average of hist (head of notes position)
                average = np.mean(hist)
                
                length_area = kwargs.get('length_area', 5)

                index_area = detect_head(non_paranada, length_area)
                
                train_info.write(i + "\n")
                # print(i)
                # print(index_area)
                # helper.show_non_paranada_plot(i, hist, non_paranada)

                # inserting feature to csv file
                for paranada in paranada_index:
                    train_data.write(str(paranada) + ", ")
                
                train_data.write(str(average) + ", ")
                train_data.write(str(index_area) + ", ")

                train_data.write(str(class_column) + "\n")

                max_num_class = kwargs.get('max_num_class', 10)

                class_counter += 1
                if class_counter == max_num_class:
                    break

            class_column += 1

        train_data.close()
        train_info.close()

    # ===============================================================

    if type == 'all' or type == 'test':
        test_data = open("test_" + identifier + ".csv", "w")
        class_column = 0
        for note in test_group:

            class_counter = 0
            for i in note:
                img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
                
                

                thresh_method = kwargs.get('thresh_method', "gaussian")
                if thresh_method == 'mean':
                    thresh_cv = cv2.ADAPTIVE_THRESH_MEAN_C
                else:
                    thresh_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

                thresh = cv2.adaptiveThreshold(img, 255, thresh_cv,
                                                cv2.THRESH_BINARY_INV, 11, 2)
                
                # calculating histogram each row of image
                hist = np.sum(thresh == 255, axis=1)
                
                # check the index row of the paranada exist
                paranada, group_paranada, paranada_index = detect_paranada(hist, i)

                # remove paranada
                non_paranada = remove_paranada(paranada, hist)
                non_paranada = remove_outlier(non_paranada)

                # calculate average of hist (head of notes position)
                average = np.mean(hist)
                
                length_area = kwargs.get('length_area', 5)

                index_area = detect_head(non_paranada, length_area)
                
                # helper.show_non_paranada_plot(i, hist, non_paranada)

                # inserting feature to csv file
                for paranada in paranada_index:
                    test_data.write(str(paranada) + ", ")

                test_data.write(str(average) + ", ")
                test_data.write(str(index_area) + ", ")

                test_data.write(str(class_column) + "\n")

                class_counter += 1

            class_column += 1

        test_data.close()

def group_data_beats():
    path = os.path.dirname(os.path.abspath(__file__))

    dataset_path = path + "\\..\\..\\img\\train_beats\\"

    whole = glob.glob1(dataset_path+"whole\\", "note-whole*")
    half = glob.glob1(dataset_path+"half\\", "note-half*")
    quarter = glob.glob1(dataset_path+"quarter\\", "note-quarter*")

    beats = [whole, half, quarter]

    test_whole = glob.glob1(dataset_path+"whole\\test\\", "note-whole*")
    test_half = glob.glob1(dataset_path+"half\\test\\", "note-half*")
    test_quarter = glob.glob1(dataset_path+"quarter\\test\\", "note-quarter*")

    test_beats = [test_whole, test_half, test_quarter]

    return beats, test_beats, dataset_path


def create_beats_csv(**kwargs):
    beats, test_beats, dataset_path = group_data_beats()

    type = kwargs.get('type', 'all')

    if type == 'all' or type == 'train':
        train_data = open("train_beats.csv", "w")
        class_column = 0
        for note in beats:

            class_counter = 0
            for i in note:
                img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)

                thresh_method = kwargs.get('thresh_method', "gaussian")
                if thresh_method == 'mean':
                    thresh_cv = cv2.ADAPTIVE_THRESH_MEAN_C
                else:
                    thresh_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

                thresh = cv2.adaptiveThreshold(img, 255, thresh_cv,
                                                cv2.THRESH_BINARY_INV, 11, 2)
                # calculating histogram each col of image
                hist_col = np.sum(thresh == 255, axis=0)
                
                for c in hist_col:
                    train_data.write(str(c) + ", ")

                train_data.write(str(class_column) + "\n")
                
                max_num_class = kwargs.get('max_num_class', 10)

                class_counter += 1
                if class_counter == max_num_class:
                    break

            class_column += 1

        train_data.close()

    # ===============================================================

    if type == 'all' or type == 'test':
        test_data = open("test_beats.csv", "w")
        class_column = 0
        for note in test_beats:

            class_counter = 0
            for i in note:
                img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
                
                

                thresh_method = kwargs.get('thresh_method', "gaussian")
                if thresh_method == 'mean':
                    thresh_cv = cv2.ADAPTIVE_THRESH_MEAN_C
                else:
                    thresh_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

                thresh = cv2.adaptiveThreshold(img, 255, thresh_cv,
                                                cv2.THRESH_BINARY_INV, 11, 2)
                
                # calculating histogram each col of image
                hist_col = np.sum(thresh == 255, axis=0)

                for c in hist_col:
                    test_data.write(str(c) + ", ")
                # test_data.write(str(min_level) + ", ")
                # test_data.write(str(max_level) + ", ")
                # test_data.write(str(average_col) + ", ")

                test_data.write(str(class_column) + "\n")

                class_counter += 1
                if class_counter >= max_num_class/4:
                    break

            class_column += 1

        test_data.close()

def noise_reduction(img):
    # convert all to float64
    img = np.float64(img)
    # create a noise of variance 25
    noise = np.random.randn(*img.shape)*10
    # Add this noise to images
    noisy = img+noise
    # Convert back to uint8
    noisy = np.uint8(np.clip(img,0,255))
    # Denoise 3rd frame considering all the 5 frames
    dst = cv2.fastNlMeansDenoising(noisy)

    return dst

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
        helper.write_log('dataset', '4', log_message)
        # helper.show_plot(i, hist, "")
    
    group_paranada = list(helper.split_tol(indices,2))
    paranada_index = [i[0] for i in group_paranada]

    if len(paranada_index) < 5:
        log_message = "FATAL ERROR: Paranada index of " + filename + " is not completely detected"
        helper.write_log('dataset', '1', log_message)
        print("Something error, please check dataset.log!")
    
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

def remove_outlier(data):
    mean = np.mean(data)
    std = np.std(data)
    stat_z = [(s-mean)/std for s in data]
    
    outlier = (np.abs(stat_z) > 2)

    return remove_paranada(outlier, data)

def detect_head(hist, length_area):
    area = 0
    index_area = 0
    for c in range(len(hist) - (length_area - 1)):
        y_vals = hist[c:c+length_area]
        this_area = helper.integrate(y_vals, (length_area - 1))
        if area < this_area:
            index_area = c + (length_area/2)
            area = this_area

    return index_area

# printable = {}
# printable["average"] = average
# printable["average_index"] = average_index
# printable["paranada"] = paranada
# printable["paranada_index"] = paranada_index
# print(printable)
# exit()