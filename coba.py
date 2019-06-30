import numpy
import cv2
from csv import reader

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


filename = 'codebooks(51.429).csv'
dataset = load_csv(filename)

img = numpy.zeros([5, 5, 3])

img[:, :, 0] = numpy.ones([5, 5])*64/255.0
img[:, :, 1] = numpy.ones([5, 5])*128/255.0
img[:, :, 2] = numpy.ones([5, 5])*192/255.0

cv2.imwrite('color_img.jpg', img)
cv2.imshow("image", img)
cv2.waitKey()
