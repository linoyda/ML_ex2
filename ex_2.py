import sys
import numpy as np


def main():

    """
    train_x, train_y -- the already known data, already classified
    test_x --- todo: Determine its labels according to the distance to train_x, train_y
    :return:
    """
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]

    # this will be relevant after we'll pick up the the value of K
    test_to_determine = np.loadtxt(test_x, delimiter=',', converters={11: lambda d: 1 if d == b'R' else 0})



def choose_k_to_knn_alg(train_x, train_y):
    training_arr = np.loadtxt(train_x, delimiter=',', converters={11: lambda f: 1 if f == b'R' else 0})
    label_arr = np.genfromtxt(train_y)

    x = training_arr[40:]
    y = label_arr[40:]

    # test x, y --- given about 10%, run knn and choose k accordingly.
    test_x = training_arr[:40]
    test_labels = label_arr[:40]


if __name__ == "__main__":
    main()
