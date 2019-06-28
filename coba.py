import os, os.path
import glob
from random import randrange
from random import seed

dataset_path = os.getcwd() + "\\img\\originals-resized"
files = os.listdir( dataset_path )
n_img = len(files)

# counting file with similar name
c1_counter = len(glob.glob1(dataset_path, "*eighth-c1*"))
print (c1_counter)

# Create a random codebook vector
def randomCodebook(trainingData):
	n_records = len(trainingData)
	n_features = len(trainingData[0])
	codebook = [trainingData[randrange(n_records)][i] for i in range(n_features)]
	return codebook

seed(1)
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

# print(dataset.pop(1))
