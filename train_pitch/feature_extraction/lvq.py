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
import numpy as np
import os
import sys
import time
import threading
import itertools
import helper

class LVQ:
    codebooks   = list()
    n_codebooks = 0
    data_train  = list()
    data_test   = list()
    is_loading  = False

    # instance attribute
    def __init__(self):
        pass

    def set_data(self, dataset, t, **kwargs):
        for i in range(len(dataset[0])-1):
            self.str_column_to_float(dataset, i)
            # self.min_max_normalize(dataset, i, 0, 255)

        # convert class column to integers
        self.str_column_to_int(dataset, -1)

        codebooks_multiplier = kwargs.get('codebooks_multiplier', 1)

        if t == 'train':
            self.data_train = dataset

            if self.n_codebooks > 0:
                self.random_codebooks(codebooks_multiplier)
        elif t == 'test':
            self.data_test = dataset
        else:
            print("Hanya menerima string 'train' atau 'test'")

    # Load a CSV file
    def load_csv(self, filename, t, **kwargs):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)

        for i in range(len(dataset[0])-1):
            self.str_column_to_float(dataset, i)
            # self.min_max_normalize(dataset, i, 0, 255)

        # convert class column to integers
        self.str_column_to_int(dataset, -1)
        
        # shuffle(dataset)

        codebooks_multiplier = kwargs.get('codebooks_multiplier', 1)

        if t == 'train':
            self.data_train = dataset

            if self.n_codebooks > 0:
                self.random_codebooks(codebooks_multiplier)

        elif t == 'test':
            self.data_test = dataset
        else:
            print("Hanya menerima string 'train' atau 'test'")

    def import_codebooks(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            n = 1
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
                n += 1

        for i in range(len(dataset[0])-1):
            self.str_column_to_float(dataset, i)

        # convert class column to integers
        self.str_column_to_int(dataset, -1)

        self.n_codebooks = n
        self.codebooks = dataset
        
        msg = "Import codebooks vector from " + filename + " success"
        helper.write_log("dataset", '3', msg)

    def export_codebooks(self, filename):
        f = open(filename + ".csv", "w")
        for codebook in self.codebooks:
            for num in codebook[:-1]:
                f.write(str(num) + ", ")
            f.write(str(codebook[-1]) + "\n")
        f.close()
        msg = "Export codebooks vector to " + filename + ".csv success"
        helper.write_log("dataset", '3', msg)

    def set_n_codebooks(self, n):
        self.n_codebooks = n

    # Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    # Convert string column to integer
    def str_column_to_int(self, dataset, column):
        for row in dataset:
            row[column] = int(row[column])

    # Normalization with min-max method
    def min_max_normalize(self, dataset, column, min=0, max=100):
        for i in range(len(dataset)):
            dataset[i][column] = round(((dataset[i][column] - min) / (max - min)), 6)

    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    # Locate the best matching unit
    def get_best_matching_unit(self, codebooks, test_row):
        distances = list()
        for codebook in codebooks:
            dist = self.euclidean_distance(codebook, test_row)
            distances.append((codebook, dist))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]

    # Make a prediction with codebook vectors
    def predict(self, codebooks, test_row):
        bmu = self.get_best_matching_unit(codebooks, test_row)
        return bmu[-1]

    # Create a random codebook vector
    def random_codebooks(self, multiplier=1):
        finded_class = list()
        codebook = list()
        for row in self.data_train:
            if finded_class.count(row[-1]) < multiplier:
                finded_class.append(row[-1])
                codebook.append(row)

            if len(finded_class) == (self.n_codebooks * multiplier):
                break
        self.codebooks = codebook

    # Train a set of codebook vectors
    def train_codebooks(self, lrate, epochs):
        if len(self.data_train) == 0:
            print("Data latih belum di input!")
            return
        
        # Loading animation
        self.is_loading = True
        thread1 = threading.Thread(target=self.animate)
        thread1.start()

        for epoch in range(epochs):
            rate = lrate * 0.1
            for row in self.data_train:
                bmu = self.get_best_matching_unit(self.codebooks, row)
                for i in range(len(row)-1):
                    error = row[i] - bmu[i]
                    if bmu[-1] == row[-1]:
                        bmu[i] += rate * error
                    else:
                        bmu[i] -= rate * error
        self.is_loading = False
        print("\nProses training selesai")

    # Calculate accuracy percentage
    def accuracy_metric(self, t='train'):
        correct = 0

        if t == 'train':
            data = self.data_train
        elif t == 'test':
            data = self.data_test
        else:
            print("Hanya menerima string 'train' atau 'test'")
            return

        actual = [row[-1] for row in data]

        predictions = list()
        for row in data:
            output = self.predict(self.codebooks, row)
            predictions.append(output)

        wrong_data = list()
        for i in range(len(actual)):
            if actual[i] == predictions[i]:
                correct += 1
            else:
                wrong_data.append(i+1)
        return (correct / float(len(actual)) * 100.0), wrong_data, actual, predictions

    def write_codebooks(self, name):
        filename = name
        f = open(filename, 'w')
        for codebook in self.codebooks:
            for val in codebook:
                f.write(str(val) + ', ')
            f.write('\n')
        f.close()

    def animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if not self.is_loading:
                sys.stdout.write('\r')
                sys.stdout.flush()
                break
            sys.stdout.write('\rMenunggu proses pembelajaran ' + c)
            sys.stdout.flush()
            time.sleep(0.1)
        print("\n")
