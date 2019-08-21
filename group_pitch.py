import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

dataset_path = path + "/img/originals-resized/"

c1 = glob.glob1(dataset_path, "*c1*")
d1 = glob.glob1(dataset_path, "*d1*")
e1 = glob.glob1(dataset_path, "*e1*")
f1 = glob.glob1(dataset_path, "*f1*")
g1 = glob.glob1(dataset_path, "*g1*")
a1 = glob.glob1(dataset_path, "*a1*")
b1 = glob.glob1(dataset_path, "*h*")
c2 = glob.glob1(dataset_path, "*c2*")
d2 = glob.glob1(dataset_path, "*d2*")
e2 = glob.glob1(dataset_path, "*e2*")
f2 = glob.glob1(dataset_path, "*f2*")
g2 = glob.glob1(dataset_path, "*g2*")
a2 = glob.glob1(dataset_path, "*a2*")
b2 = glob.glob1(dataset_path, "*h1*")
c3 = glob.glob1(dataset_path, "*c3*")

pitch = [c1,d1,e1,f1,g1, a1,b1,c2,d2,e2,f2,g2, a2,b2,c3]
#        do re mi fa sol la si do re mi fa sol la si do