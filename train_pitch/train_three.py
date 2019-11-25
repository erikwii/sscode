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
from lvq import LVQ
from group_three import *

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])

def findMiddle(input_list):
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
    
    return findMiddle(factors)

# seed(1)
learn_rate = 0.02
n_epochs = 500
n_codebooks = 5

train_beats = LVQ()
train_beats.set_n_codebooks(n_codebooks)

# load and prepare data train
filename = 'train_three.csv'
train_beats.load_csv(filename, 'train')

# load and prepare data test
filename = 'test_three.csv'
train_beats.load_csv(filename, 'test')

train_beats.train_codebooks(learn_rate, n_epochs)

print("class codebooks: ", end="")
print([row[-1] for row in train_beats.codebooks])

score, wrong_data = train_beats.accuracy_metric('train')
score_test, wrong_data_test = train_beats.accuracy_metric('test')

print("===============train==============")
print("score: " + str(score) + "%")
print("\n")
print("wrong data: ", end="")
print(wrong_data)

print("\n===============test===============")
print("score test: " + str(score_test) + "%")
print("\n")
print("wrong data test: ", end="")
print(wrong_data_test)

img_data = list()

class_column = 0
class_counter = 0
for note in pitch:
    for i in note:
        if (class_counter+1) in wrong_data:
            img = cv.imread(dataset_path + i, cv.IMREAD_GRAYSCALE)
            thresh = cv.adaptiveThreshold(
                img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
            img_data.append(thresh)
        class_counter += 1
    class_column += 1

h, w = find_middle_factor(len(img_data))
if w == 1 and len(img_data) > 10:
    h, w = find_middle_factor(len(img_data)+1)
# print(len(img_data))
# print("heigh: " + str(h) + ", width: " + str(w))
# exit()
for i in range(len(img_data)):
    plt.subplot(w, h, i+1), plt.imshow(img_data[i], 'gray')
    plt.title(wrong_data[i])
    plt.xticks([]), plt.yticks([])
plt.show()

img_data_test = list()

class_column = 0
class_counter = 0
for note in test_pitch:
    for i in note:
        if (class_counter+1) in wrong_data_test:
            img = cv.imread(dataset_path + i, cv.IMREAD_GRAYSCALE)
            thresh = cv.adaptiveThreshold(
                img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
            img_data_test.append(thresh)
        class_counter += 1
    class_column += 1

h, w = find_middle_factor(len(wrong_data_test))
if w == 1 and len(wrong_data_test) > 10:
    h, w = find_middle_factor(len(wrong_data_test)+1)
# print(len(img_data))
# exit()
for i in range(len(wrong_data_test)):
    plt.subplot(w, h, i+1), plt.imshow(img_data_test[i], 'gray')
    plt.title(wrong_data_test[i])
    plt.xticks([]), plt.yticks([])
plt.show()

exit()
for codebook in codebooks:
	print(codebook[-1])
predictions = list()
for row in test_set:
	output = predict(codebooks, row)
	predictions.append(output)

actual = [row[-1] for row in test_set]
accuracy = accuracy_metric(actual, predictions)
scores = accuracy

accuracy = str(scores)
# os.rename('codebooks.csv', 'codebooks('+accuracy+').csv')
print('\nScores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
