import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "/img/train_beats/"

whole       = glob.glob1(dataset_path+"whole/", "note-whole*")
half        = glob.glob1(dataset_path+"half/", "note-half*")
quarter     = glob.glob1(dataset_path+"quarter/", "note-quarter*")

beats = [whole, half, quarter]