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
    # ======================== TRAINING WHOLE ===================== #
    # ============================================================= #

    identifier = kwargs.get('identifier', 'quarter')

    extraction = kwargs.get('extraction', 'paranada')

    max_normalize = kwargs.get('max_normalize', 255)

    learn_rate = kwargs.get('learning_rate', 0.05)
    n_epochs = kwargs.get('max_epoch', 100)
    n_codebooks = kwargs.get('n_codebooks', 9)

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
    helper.show_wrong_data(wrong_data, predictions, pitch, dataset_path)
    # exit()

    # Show wrong data test image
    # helper.show_wrong_data(wrong_data_test, predictions_test, whole_test, dataset_path)
    # exit()

    return score, duration

# Create dataset CSV
score = list()
duration = list()
start_epoh = 1600
till_epoh = 2001
step = 200
# create_dataset.create_csv(identifier='beats', extraction='pixel', hist_axis='col', max_num_class=4, type='train')
create_dataset.create_csv(identifier='quarter', extraction='paranada', max_num_class=4, length_area=5, type='train')
for i in range(start_epoh,till_epoh,step):
    # score_i, duration_i = train_beats(extraction='pixel', learning_rate=0.03, max_epoch=i)
    score_i, duration_i = train_pitch(identifier='quarter', extraction='paranada', learning_rate=0.03, max_epoch=i)

    score.append(round(score_i,3))
    duration.append(round(duration_i,3))

print("epoh\tscore\tduration")
for i in range(len(score)):
    print(str(range(start_epoh,till_epoh,step)[i]) + "\t" + str(round(score[i],3)) + "\t" + str(duration[i]))

# show plot
# plt.subplot(1, 2, 1)
plt.plot(range(start_epoh,till_epoh,step), score)
plt.ylabel("akurasi (%)")
plt.xlabel("jumlah epoh")
plt.xticks(np.arange(start_epoh, till_epoh, step))
for i, txt in enumerate(score):
    plt.annotate(txt, (range(start_epoh,till_epoh,step)[i], score[i]))
plt.gca().yaxis.grid(True)
plt.title("Pengaruh jumlah epoh terhadap akurasi data latih")

# plt.subplot(1, 2, 2)
# x = range(30)
# plt.plot(range(50,301,50), duration, '-bo')
# plt.ylabel("durasi (detik)")
# plt.xlabel("jumlah epoh")
# plt.gca().yaxis.grid(True)
# plt.title("Pengaruh jumlah epoh terhadap running time")

plt.show()
exit()
create_dataset.create_csv(identifier='beats', extraction='histogram', hist_axis='col', max_num_class=2, type='train')
train_beats(50)

exit()
create_dataset.create_csv(identifier='whole', max_num_class=4, length_area=5, type='train')
create_dataset.create_csv(identifier='half', max_num_class=4, length_area=5, type='train')
create_dataset.create_csv(identifier='quarter', max_num_class=4, length_area=5, type='train')

# ============================================================= #
# ======================== TRAINING WHOLE ===================== #
# ============================================================= #

learn_rate = 0.05
n_epochs = 2000
n_codebooks = 9

print("learning rate: " + str(learn_rate))
print("epoch: " + str(n_epochs))
print("class: " + str(n_codebooks))
print()

train_whole = LVQ()
train_whole.set_n_codebooks(n_codebooks)

# load and prepare data train
filename = 'train_whole.csv'
train_whole.load_csv(filename, 'train')
for i in range(len(train_whole.data_train[0])-1):
    if i != 5: # difference normalization for average value
        train_whole.min_max_normalize(train_whole.data_train, i, 0, 50)
    else:
        train_whole.min_max_normalize(train_whole.data_train, i, 0, 30)

# load and prepare data test
# filename = 'test_whole.csv'
# train_whole.load_csv(filename, 'test')
# for i in range(len(train_whole.data_test[0])-1):
#     if i != 5:
#         train_whole.min_max_normalize(train_whole.data_test, i, 0, 50)
#     else:
#         train_whole.min_max_normalize(train_whole.data_test, i, 0, 30)

train_whole.train_codebooks(learn_rate, n_epochs)

print("class codebooks: ", end="")
print([row[-1] for row in train_whole.codebooks])

score, wrong_data, actual, predictions = train_whole.accuracy_metric('train')
# score_test, wrong_data_test, actual_test, predictions_test = train_whole.accuracy_metric('test')

print("===============train whole==============")
print("score: " + str(round(score, 3)) + "%")
print("\n")
print("wrong data: ", end="")
print(wrong_data)

# print("\n===============test===============")
# print("score test: " + str(round(score_test, 3)) + "%")
# print("\n")
# print("wrong data test: ", end="")
# print(wrong_data_test)

train_whole.export_codebooks("whole_codebooks")

whole, whole_test, dataset_path = create_dataset.group_data("whole")

# Show wrong data train image
# helper.show_wrong_data(wrong_data, predictions, whole, dataset_path)
# exit()

# Show wrong data test image
# helper.show_wrong_data(wrong_data_test, predictions_test, whole_test, dataset_path)
# exit()

# ============================================================= #
# ======================== TRAINING HALF ====================== #
# ============================================================= #

learn_rate = 0.03
n_epochs = 1000
n_codebooks = 9

print("learning rate: " + str(learn_rate))
print("epoch: " + str(n_epochs))
print("class: " + str(n_codebooks))
print()

train_half = LVQ()
train_half.set_n_codebooks(n_codebooks)

# load and prepare data train
filename = 'train_half.csv'
train_half.load_csv(filename, 'train')
for i in range(len(train_half.data_train[0])-1):
    if i != 5: # difference normalization for average value
        train_half.min_max_normalize(train_half.data_train, i, 0, 50)
    else:
        train_half.min_max_normalize(train_half.data_train, i, 0, 30)

# load and prepare data test
# filename = 'test_half.csv'
# train_half.load_csv(filename, 'test')
# for i in range(len(train_half.data_test[0])-1):
#     if i != 5:
#         train_half.min_max_normalize(train_half.data_test, i, 0, 50)
#     else:
#         train_half.min_max_normalize(train_half.data_test, i, 0, 30)

train_half.train_codebooks(learn_rate, n_epochs)

print("class codebooks: ", end="")
print([row[-1] for row in train_half.codebooks])

score, wrong_data, actual, predictions = train_half.accuracy_metric('train')
# score_test, wrong_data_test, actual_test, predictions_test = train_half.accuracy_metric('test')

print("===============train half==============")
print("score: " + str(round(score, 3)) + "%")
print("\n")
print("wrong data: ", end="")
print(wrong_data)

# print("\n===============test===============")
# print("score test: " + str(round(score_test, 3)) + "%")
# print("\n")
# print("wrong data test: ", end="")
# print(wrong_data_test)

train_half.export_codebooks("half_codebooks")

half, half_test, dataset_path = create_dataset.group_data("half")

# Show wrong data train image
# helper.show_wrong_data(wrong_data, predictions, half, dataset_path)
# exit()

# Show wrong data test image
# helper.show_wrong_data(wrong_data_test, predictions_test, half_test, dataset_path)
# exit()

# ============================================================= #
# ====================== TRAINING QUARTER ===================== #
# ============================================================= #

learn_rate = 0.03
n_epochs = 1000
n_codebooks = 9

print("learning rate: " + str(learn_rate))
print("epoch: " + str(n_epochs))
print("class: " + str(n_codebooks))
print()

train_quarter = LVQ()
train_quarter.set_n_codebooks(n_codebooks)

# load and prepare data train
filename = 'train_quarter.csv'
train_quarter.load_csv(filename, 'train')
for i in range(len(train_quarter.data_train[0])-1):
    if i != 5: # difference normalization for average value
        train_quarter.min_max_normalize(train_quarter.data_train, i, 0, 50)
    else:
        train_quarter.min_max_normalize(train_quarter.data_train, i, 0, 30)

# load and prepare data test
# filename = 'test_quarter.csv'
# train_quarter.load_csv(filename, 'test')
# for i in range(len(train_quarter.data_test[0])-1):
#     if i != 5:
#         train_quarter.min_max_normalize(train_quarter.data_test, i, 0, 50)
#     else:
#         train_quarter.min_max_normalize(train_quarter.data_test, i, 0, 30)

train_quarter.train_codebooks(learn_rate, n_epochs)

print("class codebooks: ", end="")
print([row[-1] for row in train_quarter.codebooks])

score, wrong_data, actual, predictions = train_quarter.accuracy_metric('train')
# score_test, wrong_data_test, actual_test, predictions_test = train_quarter.accuracy_metric('test')

print("===============train quarter==============")
print("score: " + str(round(score, 3)) + "%")
print("\n")
print("wrong data: ", end="")
print(wrong_data)

# print("\n===============test===============")
# print("score test: " + str(round(score_test, 3)) + "%")
# print("\n")
# print("wrong data test: ", end="")
# print(wrong_data_test)

train_quarter.export_codebooks("quarter_codebooks")

quarter, quarter_test, dataset_path = create_dataset.group_data("quarter")

# Show wrong data train image
helper.show_wrong_data(wrong_data, predictions, quarter, dataset_path)
# exit()

# Show wrong data test image
# helper.show_wrong_data(wrong_data_test, predictions_test, quarter_test, dataset_path)
# exit()