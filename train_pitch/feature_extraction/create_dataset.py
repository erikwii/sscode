import random
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
                counts = np.sum(thresh == 255, axis=1)
                max_hist = max(counts)
                
                # check the index row of the paranada exist
                mean = np.mean(counts)
                std = np.std(counts)
                stat_z = [(s-mean)/std for s in counts]
                paranada = (np.abs(stat_z) > 2)
                indices = [i for i, x in enumerate(paranada) if x == True]
                if indices == []:
                    paranada = (np.abs(np.abs(counts - max_hist) <= 2))
                    indices = [i for i, x in enumerate(paranada) if x == True]
                    
                    log_message = "WARNING: Failed to get outlier of " + i
                    helper.write_log('dataset', '4', log_message)
                    # helper.show_plot(i, counts, "")
                group_paranada = list(helper.split_tol(indices,2))
                paranada_index = [i[0] for i in group_paranada]
                if len(paranada_index) < 5:
                    log_message = "FATAL ERROR: Paranada index of " + i + " is not completely detected"
                    helper.write_log('dataset', '1', log_message)
                    print("Something error, please check dataset.log!")

                # remove paranada
                non_paranada = list()
                j = 0
                for x in paranada:
                    if x == True:
                        if j > 0 and j < len(paranada)-1:
                            mean = (counts[j-1]+counts[j+1])/2
                        elif j == 0:
                            mean = (counts[j+1])
                        elif j == len(paranada)-1:
                            mean = counts[j-1]
                        non_paranada.append(mean)
                    else :
                        non_paranada.append(counts[j])
                    j += 1

                # calculate average of counts (head of notes position)
                average = np.mean(counts)
                
                # mean = np.mean(non_paranada)
                # std = np.std(non_paranada)
                # stat_z = [(s - mean)/std for s in non_paranada]
                # old check
                # check = (np.abs(counts - average) <= 2)
                # check = np.abs(stat_z) > 2
                # indices = [i for i, x in enumerate(check) if x == True]
                # group_average = list(helper.split_tol(indices,2))
                # average_index = sum(indices)/len(indices)
                # print(group_average)
                # print(average_index)
                # print()
                
                length_area = kwargs.get('length_area', 5)

                area = 0
                index_area = 0
                for c in range(len(non_paranada) - (length_area - 1)):
                    y_vals = non_paranada[c:c+length_area]
                    this_area = helper.integrate(y_vals, (length_area - 1))
                    if area < this_area:
                        index_area = c + (length_area/2)
                        area = this_area
                
                # helper.show_non_paranada_plot(i, counts, non_paranada)

                # printable = {}
                # printable["average"] = average
                # printable["average_index"] = average_index
                # printable["paranada"] = paranada
                # printable["paranada_index"] = paranada_index
                # print(printable)
                # exit()

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
                counts = np.sum(thresh == 255, axis=1)
                max_hist = max(counts)
                
                # check the index row of the paranada exist
                mean = np.mean(counts)
                std = np.std(counts)
                stat_z = [(s-mean)/std for s in counts]
                paranada = (np.abs(stat_z) > 2)
                indices = [i for i, x in enumerate(paranada) if x == True]
                if indices == []:
                    paranada = (np.abs(np.abs(counts - max_hist) <= 2))
                    indices = [i for i, x in enumerate(paranada) if x == True]

                    log_message = "WARNING: Failed to get outlier of " + i
                    helper.write_log('dataset', '4', log_message)
                    # helper.show_plot(i, counts, "")
                group_paranada = list(helper.split_tol(indices,2))
                paranada_index = [i[0] for i in group_paranada]
                if len(paranada_index) < 5:
                    log_message = "FATAL ERROR: Paranada index of " + i + " is not completely detected"
                    helper.write_log('dataset', '1', log_message)
                    print("Something error, please check dataset.log!")
                
                # remove paranada
                non_paranada = list()
                j = 0
                for x in paranada:
                    if x == True:
                        if j > 0 and j < len(paranada)-1:
                            mean = (counts[j-1]+counts[j+1])/2
                        elif j == 0:
                            mean = (counts[j+1])
                        elif j == len(paranada)-1:
                            mean = counts[j-1]
                        non_paranada.append(mean)
                    else :
                        non_paranada.append(counts[j])
                    j += 1

                # calculate average of counts (head of notes position)
                average = np.mean(counts)
                
                # mean = np.mean(non_paranada)
                # std = np.std(non_paranada)
                # stat_z = [(s - mean)/std for s in non_paranada]
                # old check
                # check = (np.abs(counts - average) <= 2)
                # check = np.abs(stat_z) > 2
                # indices = [i for i, x in enumerate(check) if x == True]
                # group_average = list(helper.split_tol(indices,2))
                # average_index = sum(indices)/len(indices)
                # print(group_average)
                # print(average_index)
                # print()
                
                length_area = kwargs.get('length_area', 5)

                area = 0
                index_area = 0
                for c in range(len(non_paranada) - (length_area - 1)):
                    y_vals = non_paranada[c:c+length_area]
                    this_area = helper.integrate(y_vals, (length_area - 1))
                    if area < this_area:
                        index_area = c + (length_area/2)
                        area = this_area
                
                # helper.show_non_paranada_plot(i, counts, non_paranada)

                # printable = {}
                # printable["average"] = average
                # printable["average_index"] = average_index
                # printable["paranada"] = paranada
                # printable["paranada_index"] = paranada_index
                # print(printable)
                # exit()

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
                counts_col = np.sum(thresh == 255, axis=0)
                max_level = max(counts_col)
                min_level = min(counts_col)
                average_col = sum(counts_col)/30
                
                for c in counts_col:
                    train_data.write(str(c) + ", ")
                # train_data.write(str(min_level) + ", ")
                # train_data.write(str(max_level) + ", ")
                # train_data.write(str(average_col) + ", ")

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
                counts_col = np.sum(thresh == 255, axis=0)
                max_level = max(counts_col)
                min_level = min(counts_col)
                average_col = sum(counts_col)/30

                for c in counts_col:
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







# train_three = open("train_three.csv", "w")
# class_column = 0
# for note in pitch:

#     class_counter = 0
#     for i in note:
#         img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
#         thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                         cv2.THRESH_BINARY_INV, 11, 2)
#         # calculating histogram each row of image
#         counts = np.sum(thresh == 255, axis=1)
#         max_hist = max(counts)
#         # helper.show_plot(i,counts, "")

#         # calculate average of counts (head of notes position)
#         average = sum(counts)/50
#         check = (np.abs(counts - average) <= 2)
#         indices = [i for i, x in enumerate(check) if x == True]
#         group_average = list(helper.split_tol(ind ices,2))
#         average_index = sum(indices)/len(indices)

#         # check the index row of the paranada exist
#         paranada = (np.abs(counts - max_hist) <= 1)
#         indices = [i for i, x in enumerate(paranada) if x == True]
#         group_paranada = list(helper.split_tol(indices,2))
#         paranada_index = [i[0] for i in group_paranada]

#         # printable = {}
#         # printable["average"] = average
#         # printable["average_index"] = average_index
#         # printable["paranada"] = paranada
#         # printable["paranada_index"] = paranada_index
#         # print(printable)
#         # exit()

#         # inserting feature to csv file
#         for paranada in paranada_index:
#             train_three.write(str(paranada) + ", ")
        
#         train_three.write(str(average) + ", ")
#         train_three.write(str(average_index) + ", ")

#         train_three.write(str(class_column) + "\n")

#         class_counter += 1
#         if class_counter == 4:
#             break

#     class_column += 1

# train_three.close()

# # ===============================================================

# test_three = open("test_three.csv", "w")
# class_column = 0
# for note in test_pitch:

#     class_counter = 0
#     for i in note:
#         img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
#         thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                         cv2.THRESH_BINARY_INV, 11, 2)
        
#         # calculating histogram each row of image
#         counts = np.sum(thresh == 255, axis=1)

#         # calculate average of counts (head of notes position)
#         average = sum(counts)/50
#         check = (np.abs(counts - average) < 2)
#         indices = [i for i, x in enumerate(check) if x == True]
#         group_average = list(helper.split_tol(indices, 2))
#         average_index = sum(indices)/len(indices)

#         # check the index row of the paranada exist
#         paranada = (np.abs(counts - 30) <= 5)
#         indices = [i for i, x in enumerate(paranada) if x == True]
#         group_paranada = list(helper.split_tol(indices, 2))
#         paranada_index = [i[0] for i in group_paranada]

#         # inserting feature to csv file
#         for paranada in paranada_index:
#             test_three.write(str(paranada) + ", ")

#         test_three.write(str(average) + ", ")
#         test_three.write(str(average_index) + ", ")

#         test_three.write(str(class_column) + "\n")

#         class_counter += 1

#     class_column += 1

# test_three.close()

# exit()
