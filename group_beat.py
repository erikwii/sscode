import glob

dataset_path = "C:/Users/X230/Desktop/sscode/img/originals-resized/"

whole       = glob.glob1(dataset_path, "note-whole*")
half        = glob.glob1(dataset_path, "note-half*")
quarter     = glob.glob1(dataset_path, "note-quarter*")
eighth      = glob.glob1(dataset_path, "note-eighth*")
sixteenth   = glob.glob1(dataset_path, "note-sixteenth*")
print(whole);
beats = [whole, half, quarter]