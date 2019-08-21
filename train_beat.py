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
from group_beat import *

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])

def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return (input_list[int(middle - .5)], input_list[int(middle - .5)])
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])

def find_middle_factor(num):
    count = 0
    number = int(num)
    factors = list()
    for i in range(2, number-1):
        if number%i == 0:
            factors.append(i)
            i += 1
            count += 1
    if count==0:
        return (num, 1)

    return findMiddle(factors)

seed(1)
learn_rate = 0.01
n_epochs = 400
n_codebooks = 3

train_beats = LVQ()
train_beats.set_n_codebooks(n_codebooks)

# load and prepare data
filename = 'train_beats.csv'
train_beats.load_csv(filename, 'train')

train_beats.train_codebooks(learn_rate, n_epochs)

for c in train_beats.codebooks: print(c[-1])

score, wrong_data = train_beats.accuracy_metric('train')

print(score)
print(wrong_data)
exit()
img_data = list()

class_column = 0
for note in beats:
    class_counter = 0
    for i in note:
        if class_counter+1 in wrong_data:
            img = cv.imread(dataset_path + i, cv.IMREAD_GRAYSCALE)
            thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
            img_data.append(thresh)
        class_counter += 1
    class_column += 1

w, h = find_middle_factor(len(wrong_data))

for i in range(len(wrong_data)):
    plt.subplot(w, h, i+1), plt.imshow(img_data[i], 'gray')
    plt.title(wrong_data[i])
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
