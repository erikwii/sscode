import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "\\..\\..\\img\\eighth\\"

e1 = glob.glob1(dataset_path+"e1\\", "note-eighth*")
f1 = glob.glob1(dataset_path+"f1\\", "note-eighth*")
g1 = glob.glob1(dataset_path+"g1\\", "note-eighth*")
a1 = glob.glob1(dataset_path+"a1\\", "note-eighth*")
b1 = glob.glob1(dataset_path+"h1\\", "note-eighth*")
c2 = glob.glob1(dataset_path+"c2\\", "note-eighth*")
d2 = glob.glob1(dataset_path+"d2\\", "note-eighth*")
e2 = glob.glob1(dataset_path+"e2\\", "note-eighth*")
f2 = glob.glob1(dataset_path+"f2\\", "note-eighth*")

pitch = [e1, f1, g1, a1, b1, c2, d2, e2, f2]

test_e1 = glob.glob1(dataset_path+"e1\\test\\", "note-eighth*")
test_f1 = glob.glob1(dataset_path+"f1\\test\\", "note-eighth*")
test_g1 = glob.glob1(dataset_path+"g1\\test\\", "note-eighth*")
test_a1 = glob.glob1(dataset_path+"a1\\test\\", "note-eighth*")
test_b1 = glob.glob1(dataset_path+"h1\\test\\", "note-eighth*")
test_c2 = glob.glob1(dataset_path+"c2\\test\\", "note-eighth*")
test_d2 = glob.glob1(dataset_path+"d2\\test\\", "note-eighth*")
test_e2 = glob.glob1(dataset_path+"e2\\test\\", "note-eighth*")
test_f2 = glob.glob1(dataset_path+"f2\\test\\", "note-eighth*")

test_pitch = [test_e1, test_f1, test_g1, test_a1, test_b1, test_c2, test_d2, test_e2, test_f2]