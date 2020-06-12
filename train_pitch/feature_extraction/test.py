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

# ==========
# TEST BEATS
# ==========
create_dataset.create_csv_test(max_num_class=0, length_area=7)

learning_rate = 0.05
max_epoh = 150
n_codebooks = 3

test_beats = LVQ()
test_beats.set_n_codebooks(n_codebooks)

# load and prepare data test
filename = 'beats_codebooks.csv'
test_beats.import_codebooks(filename)
test_beats.load_csv("test_histogram.csv", "test")

# for i in range(len(test_beats.data_test[0])-1):
#     test_beats.min_max_normalize(test_beats.data_test, i, 0, 255)

score_beats, wrong_data_beats, actual_beats, predictions_beats = test_beats.accuracy_metric('test')

# ==========
# TEST PITCH
# ==========
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

score_pitch, wrong_data_pitch, actual_pitch, predictions_pitch = test_pitch.accuracy_metric('test')

# =========
# Statistic
# =========

pitch_info = open("test_paranada_info.csv", 'r')
pitch_data = pitch_info.read()
pitch_data = pitch_data.split("\n")
pitch_data.pop()

file_info = open("test_histogram_info.csv", 'r')
data_info = file_info.read()
data_info = data_info.split("\n")
data_info.pop()

pitch_arranged = list()
for i in range(len(data_info)):
    for j in range(len(pitch_data)):
        if data_info[i] == pitch_data[j]:
            pitch_arranged.append([actual_pitch[j], predictions_pitch[j]])

beat_class = ['whole', 'half', 'quarter']
pitch_class = ['e1', 'f1', 'g1', 'a1', 'b1', 'c2', 'd2', 'e2', 'f2']

# Printing accuracy table
skor_all = 0
print("\nNama\t\t\t\tBeat asli\tPitch asli\tBeat prediksi\tPitch prediksi\tSkor")
for i in range(len(actual_beats)):
    print(data_info[i], end="")
    
    if len(data_info[i]) < 24:
        print("\t", end="")
    
    print("\t", end="")
    print(beat_class[actual_beats[i]], end="")
    
    print("\t\t", end="")
    print(pitch_class[pitch_arranged[i][0]], end="")
    
    print("\t\t", end="")
    print(beat_class[predictions_beats[i]], end="")
    
    print("\t\t", end="")
    print(pitch_class[pitch_arranged[i][1]], end="")
    
    skor = 0
    if actual_beats[i] == predictions_beats[i]:
        skor += 1
    if pitch_arranged[i][0] == pitch_arranged[i][1]:
        skor += 1
    if skor == 2:
        skor_all += 1
    print("\t\t", end="")
    print(skor)

print("\n====== TEST BEATS ======")
print("jumlah data: " + str(len(predictions_beats)))
print("benar: " + str(len(actual_beats) - len(wrong_data_beats)) + "\tsalah: " + str(len(wrong_data_beats)))
print("score: " + str(round(score_beats, 3)) + "%")
print()
print("====== TEST PITCH ======")
print("jumlah data: " + str(len(predictions_pitch)))
print("benar: " + str(len(actual_pitch) - len(wrong_data_pitch)) + "\tsalah: " + str(len(wrong_data_pitch)))
print("score: " + str(round(score_pitch, 3)) + "%")

print("\nSKOR AKURASI GABUNGAN: " + str(round(skor_all/len(actual_beats)*100, 3)) + "%", end=" ")
print("(benar: " + str(skor_all) + "\tsalah: " + str(len(actual_beats)-skor_all) + ")")
print("=============================")