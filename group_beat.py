import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "/img/originals-resized/"

whole       = glob.glob1(dataset_path, "note-whole*")
half        = glob.glob1(dataset_path, "note-half*")
quarter     = glob.glob1(dataset_path, "note-quarter*")
eighth      = glob.glob1(dataset_path, "note-eighth*")
sixteenth   = glob.glob1(dataset_path, "note-sixteenth*")

beats = [whole, half, quarter]