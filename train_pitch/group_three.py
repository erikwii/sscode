import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "/../img/quarter/"

e1 = glob.glob1(dataset_path+"e1/", "note-quarter*")
f1 = glob.glob1(dataset_path+"f1/", "note-quarter*")
g1 = glob.glob1(dataset_path+"g1/", "note-quarter*")
a1 = glob.glob1(dataset_path+"a1/", "note-quarter*")
b1 = glob.glob1(dataset_path+"h1/", "note-quarter*")

pitch = [e1, f1, g1, a1, b1]

test_e1 = glob.glob1(dataset_path+"e1/test/", "note-quarter*")
test_f1 = glob.glob1(dataset_path+"f1/test/", "note-quarter*")
test_g1 = glob.glob1(dataset_path+"g1/test/", "note-quarter*")
test_a1 = glob.glob1(dataset_path+"a1/test/", "note-quarter*")
test_b1 = glob.glob1(dataset_path+"h1/test/", "note-quarter*")

test_pitch = [test_e1, test_f1, test_g1, test_a1, test_b1]
