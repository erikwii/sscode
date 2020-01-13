import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "\\..\\img\\train_beats\\"

whole = glob.glob1(dataset_path, "note-whole*")
half = glob.glob1(dataset_path, "note-half*")
quarter = glob.glob1(dataset_path, "note-quarter*")
eighth = glob.glob1(dataset_path, "note-eighth*")

beats = [whole, half, quarter, eighth]

test_whole = glob.glob1(dataset_path+"test\\", "note-whole*")
test_half = glob.glob1(dataset_path+"test\\", "note-half*")
test_quarter = glob.glob1(dataset_path+"test\\", "note-quarter*")
test_eighth = glob.glob1(dataset_path+"test\\", "note-eighth*")

test_beats = [test_whole, test_half, test_quarter, test_eighth]
