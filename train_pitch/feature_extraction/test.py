"""
Classification of Music Notation Image using Learning Vector Quantization (LVQ) Method
By: Erik Santiago
github: @erikwii  instagram: @erikwiii
"""

from random import seed
from random import randrange
from random import shuffle
from csv import reader
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os
import sys
import time
import threading
import itertools
import create_dataset
import helper
from lvq import LVQ

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])

def find_middle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return (input_list[int(middle + .5)], input_list[int(middle - .5)])
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])

def find_middle_factor(num):
    count = 0
    number = int(num)
    factors = list()
    for i in range(2, number-1):
        if number % i == 0:
            factors.append(i)
            i += 1
            count += 1
    
    if count == 0:
        return (num, 1)

    if count == 1:
        return (factors[0], factors[0])
    
    return find_middle(factors)

def train_beats(**kwargs):
    # ============================================================= #
    # ======================== TRAINING BEATS ===================== #
    # ============================================================= #

    extraction = kwargs.get('extraction', 'pixel')

    max_normalize = kwargs.get('max_normalize', 255)

    learn_rate = kwargs.get('learning_rate', 0.05)
    n_epochs = kwargs.get('max_epoch', 100)
    n_codebooks = kwargs.get('n_codebooks', 3)

    print("learning rate: " + str(learn_rate))
    print("epoch: " + str(n_epochs))
    print("class: " + str(n_codebooks))
    print()

    train_beats = LVQ()
    train_beats.set_n_codebooks(n_codebooks)

    # load and prepare data train
    filename = 'train_beats.csv'
    train_beats.load_csv(filename, 'train')
    if extraction == 'pixel':
        for i in range(len(train_beats.data_train[0])-1):
            train_beats.min_max_normalize(train_beats.data_train, i, 0, 255)
    else:
        for i in range(len(train_beats.data_train[0])-1):
            train_beats.min_max_normalize(train_beats.data_train, i, 0, 50)

    # Training process
    start_time = time.time()
    train_beats.train_codebooks(learn_rate, n_epochs)
    duration = time.time() - start_time

    print("\nclass codebooks: ", end="")
    print([row[-1] for row in train_beats.codebooks])

    score, wrong_data, actual, predictions = train_beats.accuracy_metric('train')

    print("===============train beats==============")
    print("Waktu proses pembelajaran: %s detik ---" % (duration))
    print("score: " + str(round(score, 3)) + "%\n")
    print("wrong data: ", end="")
    print(wrong_data)

    train_beats.export_codebooks("beats_codebooks")

    beats, beats_test, dataset_path = create_dataset.group_data_beats()

    # Show wrong data image
    # helper.show_wrong_data(wrong_data, predictions, beats, dataset_path)
    # exit()

    return score, duration

def train_pitch(**kwargs):
    # ============================================================= #
    # ======================== TRAINING PITCH ===================== #
    # ============================================================= #

    identifier = kwargs.get('identifier', 'quarter')

    extraction = kwargs.get('extraction', 'paranada')

    max_normalize = kwargs.get('max_normalize', 255)

    learn_rate = kwargs.get('learning_rate', 0.05)
    n_epochs = kwargs.get('max_epoch', 100)
    n_codebooks = kwargs.get('n_codebooks', 9)

    show_wrong_data = kwargs.get('show_wrong_data', False)

    print("learning rate: " + str(learn_rate))
    print("epoch: " + str(n_epochs))
    print("class: " + str(n_codebooks))
    print()

    train_pitch = LVQ()
    train_pitch.set_n_codebooks(n_codebooks)

    # load and prepare data train
    filename = 'train_'+ identifier +'.csv'
    train_pitch.load_csv(filename, 'train')
    if extraction == 'paranada':
        for i in range(len(train_pitch.data_train[0])-1):
            if i != 5: # difference normalization for average value
                train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 50)
            else:
                train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 30)
    elif extraction == 'pixel':
        for i in range(len(train_pitch.data_train[0])-1):
            train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 255)
    else:
        for i in range(len(train_pitch.data_train[0])-1):
            train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 30)

    # load and prepare data test
    # filename = 'test_whole.csv'
    # train_pitch.load_csv(filename, 'test')
    # for i in range(len(train_pitch.data_test[0])-1):
    #     if i != 5:
    #         train_pitch.min_max_normalize(train_pitch.data_test, i, 0, 50)
    #     else:
    #         train_pitch.min_max_normalize(train_pitch.data_test, i, 0, 30)

    # Training process
    start_time = time.time()
    train_pitch.train_codebooks(learn_rate, n_epochs)
    duration = time.time() - start_time

    print("class codebooks: ", end="")
    print([row[-1] for row in train_pitch.codebooks])

    score, wrong_data, actual, predictions = train_pitch.accuracy_metric('train')
    # score_test, wrong_data_test, actual_test, predictions_test = train_pitch.accuracy_metric('test')

    print("===============train "+ identifier +"==============")
    print("score: " + str(round(score, 3)) + "%")
    print("\n")
    print("wrong data: ", end="")
    print(wrong_data)

    # print("\n===============test===============")
    # print("score test: " + str(round(score_test, 3)) + "%")
    # print("\n")
    # print("wrong data test: ", end="")
    # print(wrong_data_test)

    train_pitch.export_codebooks(identifier +"_codebooks")

    pitch, pitch_test, dataset_path = helper.get_dataset_info(identifier, "train")
    # print(pitch)
    # print()
    # print(pitch_test)
    # print()
    # print(dataset_path)
    # exit()

    # Show wrong data train image
    if show_wrong_data:
        helper.show_wrong_data(wrong_data, predictions, pitch, dataset_path)
    # exit()

    # Show wrong data test image
    # helper.show_wrong_data(wrong_data_test, predictions_test, whole_test, dataset_path)
    # exit()

    return score, duration

create_dataset.create_csv_test(max_num_class=0, length_area=7)

learning_rate = 0.05
max_epoh = 150
n_codebooks = 3

test_beats = LVQ()
test_beats.set_n_codebooks(n_codebooks)

# load and prepare data test
filename = 'beats_codebooks.csv'
test_beats.import_codebooks(filename)
test_beats.load_csv("test_pixel.csv", "test")

for i in range(len(test_beats.data_test[0])-1):
    test_beats.min_max_normalize(test_beats.data_test, i, 0, 255)

# for i in range(len(test_beats.data_test)):
#     print(test_beats.data_test[i][-1])

for i in range(len(test_beats.codebooks[0])-1):
    test_beats.min_max_normalize(test_beats.codebooks, i, 0, 255)

score, wrong_data, actual, predictions = test_beats.accuracy_metric('test')

print("score: " + str(round(score, 3)) + "%")
print("\n")
print("wrong data: ", end="")
print(wrong_data)
print("Actual: ", end="")
print(actual)
print("Prediction: ", end="")
print(predictions)



create_dataset.create_csv_test(max_num_class=0, length_area=7)

learning_rate = 0.05
max_epoh = 4000
n_codebooks = 9

test_pitch = LVQ()
test_pitch.set_n_codebooks(n_codebooks)

# load and prepare data test
filename = 'all_codebooks.csv'
test_pitch.import_codebooks(filename)
test_pitch.load_csv("test_paranada.csv", "test")

for i in range(len(test_pitch.data_test[0])-1):
    test_pitch.min_max_normalize(test_pitch.data_test, i, 0, 50)

for i in range(len(test_pitch.data_test)):
    print(test_pitch.data_test[i])

for i in range(len(test_pitch.codebooks[0])-1):
    test_pitch.min_max_normalize(test_pitch.codebooks, i, 0, 50)

score, wrong_data, actual, predictions_p = test_pitch.accuracy_metric('test')

print("score: " + str(round(score, 3)) + "%")
print("\n")
print("wrong data: ", end="")
print(wrong_data)
print("Actual: ", end="")
print(actual)
print("Prediction: ", end="")
print(predictions_p)