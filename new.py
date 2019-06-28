import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from grouping import *

bimbelx = open("bimbelx.csv", "w")
class_column = 0
for note in c1:
    class_counter = 0
    for i in note:
        img = cv2.imread(dataset_path + i, cv2.IMREAD_GRAYSCALE)
        for row in img:
            for column in row:
                bimbelx.write(str(column) + ", ")
        bimbelx.write(str(class_column) + "\n")
        class_counter += 1
        if class_counter == 10:
            break
    class_column += 1
bimbelx.close()
