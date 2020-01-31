from matplotlib import pyplot as plt

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

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])


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

# Showing plot
def show_plot(img_name, counts, counts_col=""):
    y = range(49, -1, -1)
    plt.subplot(1, 2, 1)
    plt.plot(counts, y)
    plt.title(img_name + ' (Row)')

    if counts != "":
        plt.subplot(1, 2, 2)
        x = range(30)
        plt.plot(x, counts_col)
        plt.title(img_name + ' (Col)')

    plt.show()