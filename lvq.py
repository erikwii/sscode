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

# def animate():
# 	for c in itertools.cycle(['|', '/', '-', '\\']):
# 		if done:
# 			sys.stdout.write('\r')
# 			sys.stdout.flush()
# 			break
# 		sys.stdout.write('\rmenunggu proses pembelajaran ' + c)
# 		sys.stdout.flush()
# 		time.sleep(0.1)
# 	sys.stdout.write('\rDone!     ')

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

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, *args):
	scores = list()
	
	train_set = list(dataset)
	print(len(train_set))
	train_set = sum(train_set, [])
	print(len(train_set))
	exit()
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		test_set.append(row_copy)
		row_copy[-1] = nil
	
	print(len(dataset))
	print(len(train_set))
	print(len(test_set))
	exit()
	train = open("train.csv", 'w')
	test = open("test.csv", 'w')
	
	for tr in train_set:
		train.write(str(tr)+"\n")
	train.close()

	for ts in test_set:
		test.write(str(ts)+"\n")
	test.close()
	
	exit()

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
test_set = load_csv("test.csv")
shuffle(dataset)
shuffle(test_set)
for i in range(len(dataset[0])-1):
	str_column_to_int(dataset, i)

for i in range(len(dataset[0])-1):
	min_max_normalize(dataset, i, 0, 255)

# convert class column to integers
str_column_to_int(dataset, -1)

for i in range(len(test_set[0])-1):
	str_column_to_int(test_set, i)

for i in range(len(test_set[0])-1):
	min_max_normalize(test_set, i, 0, 255)

# convert class column to integers
str_column_to_int(test_set, -1)

# evaluate algorithm
n_folds = 2
learn_rate = 0.05
n_epochs = 200
n_codebooks = 3

done = False
# Loading animation
# thread1 = threading.Thread(target = animate)
# thread1.start()

# scores = evaluate_algorithm(
# 	dataset, learning_vector_quantization, n_codebooks, learn_rate, n_epochs)
done = True

codebooks = train_codebooks(dataset, n_codebooks, learn_rate, n_epochs)
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
