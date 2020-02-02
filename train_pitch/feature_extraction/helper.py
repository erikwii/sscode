import cv2 as cv
from matplotlib import pyplot as plt
from datetime import datetime

# helper generator
def split_tol(test_list, tol):
    res = []
    last = test_list[0]
    for ele in test_list:
        if ele-last > tol:
            yield res
            res = []
        res.append(ele)
        last = ele
    yield res

def find_paranada_index(paranada_list, average_list):
    max = 0
    index = -1
    for i in range(0, 5):
        for j in average_list:
            for n in j:
                paranada_np = np.asarray(paranada_list[i])
                paranada = (np.abs(paranada_np - n) <= 2)
                count = paranada.tolist().count(True)
                if count > max:
                    max = count
                    index = i+1
    return index

# find area of under curve
def integrate(y_vals, h):
    i = 1
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i%2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h/3.0)

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])


def standar_deviation(array):
    for i in range(len(array)):
        array[i]=int(array[i])
    jumlah=0
    for i in range(len(array)):
        jumlah +=array[i]
    rata=jumlah/len(array)
    sigma = 0
    for i in range(len(array)):
        hitung =(array[i]-rata)**2
        sigma += hitung
    pembagianN=sigma/len(array)
    standarDeviasi=pembagianN ** 0.5
    
    return standarDeviasi

def find_middle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return (input_list[int(middle + .5)], input_list[int(middle - .5)])
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])


def find_middle_factor(num):
    count = 0
    number = int(num)
    factors = list()
    for i in range(2, number-1):
        if number % i == 0:
            factors.append(i)
            i += 1
            count += 1

    if count == 0:
        return (num, 1)

    if count == 1:
        return (factors[0], factors[0])

    return find_middle(factors)

def show_wrong_data(wrong_data, predictions, dataset, dataset_path):
    img_data = list()
    notes = list()

    class_column = 0
    class_counter = 0
    for note in dataset:
        for i in note:
            if (class_counter+1) in wrong_data:
                img = cv.imread(dataset_path + i, cv.IMREAD_GRAYSCALE)
                thresh = cv.adaptiveThreshold(
                    img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
                img_data.append(thresh)
                filename = i.split("-")
                notes.append(filename[-2] + "-" + filename[-1] + " (" + str(predictions[class_counter]) + ")")
            class_counter += 1
        class_column += 1

    h, w = find_middle_factor(len(img_data))
    if w == 1 and len(img_data) > 10:
        h, w = find_middle_factor(len(img_data)+1)
    # print(len(img_data))
    # print("heigh: " + str(h) + ", width: " + str(w))
    # exit()
    for i in range(len(img_data)):
        plt.subplot(w, h, i+1), plt.imshow(img_data[i], 'gray')
        plt.title(notes[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# Showing plot
def show_plot(img_name, counts, counts_col=""):
    y = range(49, -1, -1)
    plt.subplot(1, 2, 1)
    plt.plot(counts, y)
    plt.title(img_name + ' (Row)')

    if counts_col != "":
        plt.subplot(1, 2, 2)
        x = range(30)
        plt.plot(x, counts_col)
        plt.title(img_name + ' (Col)')

    plt.show()

def show_non_paranada_plot(img_name, counts, non_paranada):
    y = range(49, -1, -1)
    plt.subplot(1, 2, 1)
    plt.plot(counts, y)
    plt.axis((0,30,0,50))
    plt.title(img_name + ' (paranada)')
    
    # y = range(49, -1, -1)
    plt.subplot(1, 2, 2)
    plt.plot(non_paranada, y)
    plt.axis((0,30,0,50))
    plt.title(img_name + ' (non paranada)')

    plt.show()

def write_log(type, level, message):
    f = open(type+".log", 'a+')
    
    timestamp = datetime.now().timestamp()
    dt_object = datetime.fromtimestamp(timestamp).strftime("%m/%d/%Y, %H:%M:%S")
    
    f.write("["+dt_object+"] (level: " +  level + ') ' + message + "\n")
    f.close()