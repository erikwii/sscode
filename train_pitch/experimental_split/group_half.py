import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "\\..\\..\\img\\half\\"

e1 = glob.glob1(dataset_path+"e1\\", "note-half*")
f1 = glob.glob1(dataset_path+"f1\\", "note-half*")
g1 = glob.glob1(dataset_path+"g1\\", "note-half*")
a1 = glob.glob1(dataset_path+"a1\\", "note-half*")
b1 = glob.glob1(dataset_path+"h1\\", "note-half*")
c2 = glob.glob1(dataset_path+"c2\\", "note-half*")
d2 = glob.glob1(dataset_path+"d2\\", "note-half*")
e2 = glob.glob1(dataset_path+"e2\\", "note-half*")
f2 = glob.glob1(dataset_path+"f2\\", "note-half*")

pitch = [e1, f1, g1, a1, b1, c2, d2, e2, f2]

test_e1 = glob.glob1(dataset_path+"e1\\test\\", "note-half*")
test_f1 = glob.glob1(dataset_path+"f1\\test\\", "note-half*")
test_g1 = glob.glob1(dataset_path+"g1\\test\\", "note-half*")
test_a1 = glob.glob1(dataset_path+"a1\\test\\", "note-half*")
test_b1 = glob.glob1(dataset_path+"h1\\test\\", "note-half*")
test_c2 = glob.glob1(dataset_path+"c2\\test\\", "note-half*")
test_d2 = glob.glob1(dataset_path+"d2\\test\\", "note-half*")
test_e2 = glob.glob1(dataset_path+"e2\\test\\", "note-half*")
test_f2 = glob.glob1(dataset_path+"f2\\test\\", "note-half*")

test_pitch = [test_e1, test_f1, test_g1, test_a1, test_b1, test_c2, test_d2, test_e2, test_f2]