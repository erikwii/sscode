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

def train_beats(**kwargs):
    # ============================================================= #
    # ======================== TRAINING BEATS ===================== #
    # ============================================================= #

    extraction = kwargs.get('extraction', 'pixel')

    max_normalize = kwargs.get('max_normalize', 255)

    learn_rate = kwargs.get('learning_rate', 0.05)
    n_epochs = kwargs.get('max_epoch', 100)
    n_codebooks = kwargs.get('n_codebooks', 3)
    codebooks_multiplier = kwargs.get('codebooks_multiplier', 1)

    print("learning rate: " + str(learn_rate))
    print("epoch: " + str(n_epochs))
    print("class: " + str(n_codebooks))
    print()

    train_beats = LVQ()
    train_beats.set_n_codebooks(n_codebooks)

    # load and prepare data train
    filename = 'train_beats.csv'
    train_beats.load_csv(filename, 'train', codebooks_multiplier=codebooks_multiplier)
    if extraction == 'pixel':
        for i in range(len(train_beats.data_train[0])-1):
            train_beats.min_max_normalize(train_beats.data_train, i, 0, 255)
    
    train_beats.random_codebooks(codebooks_multiplier)
    # print([item[-1] for item in train_beats.codebooks])
    # exit()
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
    codebooks_multiplier = kwargs.get('codebooks_multiplier', 1)

    show_wrong_data = kwargs.get('show_wrong_data', False)

    print("\n\nlearning rate: " + str(learn_rate))
    print("epoch: " + str(n_epochs))
    print("class: " + str(n_codebooks))
    print()

    train_pitch = LVQ()
    train_pitch.set_n_codebooks(n_codebooks)

    # load and prepare data train
    filename = 'train_'+ identifier +'.csv'
    train_pitch.load_csv(filename, 'train', codebooks_multiplier=codebooks_multiplier)
    if extraction == 'paranada':
        for i in range(len(train_pitch.data_train[0])-1):
            train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 50)
    elif extraction == 'pixel':
        for i in range(len(train_pitch.data_train[0])-1):
            train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 255)
    else:
        for i in range(len(train_pitch.data_train[0])-1):
            train_pitch.min_max_normalize(train_pitch.data_train, i, 0, 30)
    
    train_pitch.random_codebooks(2)

    # Training process
    start_time = time.time()
    train_pitch.train_codebooks(learn_rate, n_epochs)
    duration = time.time() - start_time

    print("class codebooks: ", end="")
    print([row[-1] for row in train_pitch.codebooks])

    score, wrong_data, actual, predictions = train_pitch.accuracy_metric('train')

    print("===============train "+ identifier +"==============")
    print("score: " + str(round(score, 3)) + "%")
    print("\n")
    print("wrong data: ", end="")
    print(wrong_data)

    train_pitch.export_codebooks(identifier +"_codebooks")

    pitch, pitch_test, dataset_path = helper.get_dataset_info(identifier, "train")

    # Show wrong data train image
    if show_wrong_data:
        helper.show_wrong_data(wrong_data, predictions, pitch, dataset_path)

    return score, duration

# ================ Train Beats ================ #
create_dataset.create_csv(identifier='beats', extraction='histogram', hist_axis='col', max_num_class=36, type='train')
score_beats, duration_beats = train_beats(extraction='histogram', codebooks_multiplier=3, learning_rate=0.1, max_epoch=3000, show_wrong_data=False)
exit()
# ================ Train Pitch ================= #
create_dataset.create_csv(identifier='all', extraction='paranada', hist_axis='row', max_num_class=16, length_area=7, type='train')
score_pitch, duration_pitch = train_pitch(identifier='all', extraction='paranada', learning_rate=0.03, max_epoch=5000, show_wrong_data=False)

print("\nScore Beats:\t" + str(round(score_beats, 2)) + "% (" + str(round(duration_beats, 2)) + "s)")
print("\nScore Pitch:\t" + str(round(score_pitch, 2)) + "% (" + str(round(duration_pitch, 2)) + "s)")