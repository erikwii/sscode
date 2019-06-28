import glob

dataset_path = "C:/Users/X230/Desktop/sscode/img/originals-resized/"
# Group C1
whole_c1 = glob.glob1(dataset_path, "*whole-c1*")
half_c1 = glob.glob1(dataset_path, "*half-c1*")
quarter_c1 = glob.glob1(dataset_path, "*quarter-c1*")
eighth_c1 = glob.glob1(dataset_path, "*eighth-c1*")
sixteenth_c1 = glob.glob1(dataset_path, "*sixteenth-c1*")

c1 = [whole_c1, half_c1, quarter_c1, eighth_c1, sixteenth_c1]
# # Group D1
# whole_d1 = glob.glob1(dataset_path, "*whole-d1*")
# half_d1 = glob.glob1(dataset_path, "*half-d1*")
# quarter_d1 = glob.glob1(dataset_path, "*quarter-d1*")
# eighth_d1 = glob.glob1(dataset_path, "*eighth-d1*")
# sixteenth_d1 = glob.glob1(dataset_path, "*sixteenth-d1*")
