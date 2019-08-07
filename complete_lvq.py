# LVQ for the Ionosphere Dataset
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

def animate():
	for c in itertools.cycle(['|', '/', '-', '\\']):
		if done:
			sys.stdout.write('\r')
			sys.stdout.flush()
			break
		sys.stdout.write('\rmenunggu proses pembelajaran ' + c)
		sys.stdout.flush()
		time.sleep(0.1)
	sys.stdout.write('\rDone!     ')

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])

# Normalization with min-max method
def min_max_normalize(dataset, column, min=0, max=100):
	for i in range(len(dataset)):
		dataset[i][column] = round((dataset[i][column] - min) / (max - min), 6)

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]

# Create a random codebook vector
def random_codebook(train, n_codebooks):
	finded_class = list()
	codebook = list()
	for row in train:
		if (row[-1] in finded_class) == False:
			finded_class.append(row[-1])
			codebook.append(row)

		if len(finded_class) == n_codebooks:
			break
	return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = random_codebook(train, n_codebooks)

	for epoch in range(epochs):
		rate = lrate * 0.1
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return codebooks

# LVQ Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	write_codebooks(codebooks)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)
	return(predictions)

def write_codebooks(codebooks):
	filename = 'codebooks.csv'
	f = open(filename, 'w')
	for codebook in codebooks:
		for val in codebook:
			f.write(str(val) + ', ')
		f.write('\n')
	f.close()

seed(1)

# load and prepare data
filename = 'bimbelx.csv'
dataset = load_csv(filename)
shuffle(dataset)
for i in range(len(dataset[0])-1):
	str_column_to_int(dataset, i)

for i in range(len(dataset[0])-1):
	min_max_normalize(dataset, i, 0, 255)

# convert class column to integers
str_column_to_int(dataset, -1)

# evaluate algorithm
n_folds = 5
learn_rate = 0.05
n_epochs = 200
n_codebooks = 4

done = False
# Loading animation
thread1 = threading.Thread(target = animate)
thread1.start()

scores = evaluate_algorithm(dataset, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
done = True

accuracy = str(round(sum(scores)/float(len(scores)), 3))
os.rename('codebooks.csv', 'codebooks('+accuracy+').csv')
print('\nScores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))