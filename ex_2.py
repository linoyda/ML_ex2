import sys
import numpy as np


def main():
    """
    train_x, train_y -- the already known data, already classified
    test_x --- todo: Determine its labels according to the distance to train_x, train_y
    """

    total_iterations = 12
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    # choose_k_to_knn_alg(train_x, train_y)

    # this will be relevant after choosing the value of K (after we determined its value using choose_k_to_knn_alg)
    test_to_determine = np.loadtxt(test_x, delimiter=',', converters={11: lambda d: 1 if d == b'R' else 0})
    training_arr = np.loadtxt(train_x, delimiter=',', converters={11: lambda f: 1 if f == b'R' else 0})
    label_arr = np.genfromtxt(train_y)

    # inspiration for Min-Max normalization:
    # https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79

    # training set...
    for count in range(total_iterations):
        training_arr[:, count] = (training_arr[:, count] - training_arr[:, count].min()) /\
                                 (training_arr[:, count].max() - training_arr[:, count].min())

    # test set...
    for count in range(total_iterations):
        test_to_determine[:, count] = (test_to_determine[:, count] - test_to_determine[:, count].min()) /\
                                      (test_to_determine[:, count].max() - test_to_determine[:, count].min())
    best_k = 15
    """
    for _k in range(3, 19, 2):
        print("k: " + str(_k) + "\n")
        alg_labels = knn_algorithm_implementation(test_to_determine, _k, training_arr, label_arr)
        print(alg_labels)
    """
    alg_labels = knn_algorithm_implementation(test_to_determine, best_k, training_arr, label_arr)
    print(alg_labels)

    # **** End of KNN part. Now - Driver code for Perceptron.
    perceptron_output = train_multiclass_perceptron(train_x, train_y, test_x)
    print(perceptron_output)


# This function determines the value of k that is preferable to the KNN implementation.
# I've come to a conclusion that k = 11, 15 gives us an accuracy percentage of 82.92682926829268% ( >> 80%)
def choose_k_to_knn_alg(train_x, train_y):
    min_k_checked = 3
    max_k_checked = 18
    skip_even = 2
    threshold = 41
    accuracy_list = []

    training_arr = np.loadtxt(train_x, delimiter=',', converters={11: lambda f: 1 if f == b'R' else 0})
    label_arr = np.genfromtxt(train_y)

    x = training_arr[threshold:]
    y = label_arr[threshold:]

    # test x, y --- given about 10% (of total examples), run knn and choose k accordingly.
    test_x = training_arr[:threshold]
    labels_to_compare = label_arr[:threshold]

    total_iterations = 12
    # training set...
    for count in range(total_iterations):
        x[:, count] = (x[:, count] - x[:, count].min()) / (x[:, count].max() - x[:, count].min())

    # test set...
    for count in range(total_iterations):
        test_x[:, count] =\
            (test_x[:, count] - test_x[:, count].min()) / (test_x[:, count].max() - test_x[:, count].min())

    for curr_k in range(min_k_checked, max_k_checked, skip_even):
        alg_labels = knn_algorithm_implementation(test_x, curr_k, x, y)

        # After we got the labels, let's compare them to the REAL ones.
        sum_of_matches = 0
        size_of_labels_list = len(alg_labels)
        for j in range(size_of_labels_list):
            if labels_to_compare[j] != alg_labels[j]:
                continue
            else:
                sum_of_matches += 1
        curr_accuracy_percentage = (sum_of_matches / size_of_labels_list) * 100
        accuracy_list.append(curr_accuracy_percentage)
        print("{} : {}".format(curr_k, accuracy_list[-1]))


def knn_algorithm_implementation(test_set, curr_k, x, y):
    test_len = len(test_set)
    x_len = len(x)
    results = []
    for count_i in range(test_len):
        distances_set = []
        labels_of_adj = []
        for count_j in range(x_len):
            curr_distance = np.linalg.norm(test_set[count_i] - x[count_j])
            distances_set.append(curr_distance)
        k_closest_adj = np.argpartition(distances_set, curr_k)

        # Determine what are the neighbors' classes, and according to the majority - determine the class
        for count_curr_k in range(curr_k):
            curr_label = y[k_closest_adj[count_curr_k]]
            labels_of_adj.append(curr_label)
        # Finding the class according to the majority of neighbors' labels.
        num_of_occurrences = np.bincount(labels_of_adj)

        # Add the class of the majority to the results list
        results.append(np.argmax(num_of_occurrences))
    return results


def train_multiclass_perceptron(x_train, y_train, test_x):
    perceptron_output = []
    eta = 0.3611
    total_iterations = 12
    col_amount = 13
    size = 355
    temp_list = [1]*355
    total_epoches = 100

    x_train = np.loadtxt(x_train, delimiter=',', converters={11: lambda f: 1 if f == b'R' else 0})
    y_train = np.genfromtxt(y_train)
    test_to_determine = np.loadtxt(test_x, delimiter=',', converters={11: lambda d: 1 if d == b'R' else 0})

    # This is a multiclass perceptron. So, we need a matrix-shaped W. 3 lines - for each class.
    w = [np.zeros(col_amount), np.zeros(col_amount), np.zeros(col_amount)]

    for index in range(total_iterations):  # Min-Max normalization
        x_train[:, index] = \
            (x_train[:, index] - x_train[:, index].min()) / (x_train[:, index].max() - x_train[:, index].min())

    for count in range(total_iterations):
        test_to_determine[:, count] = (test_to_determine[:, count] - test_to_determine[:, count].min()) /\
                                      (test_to_determine[:, count].max() - test_to_determine[:, count].min())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    test_to_determine = np.array(test_to_determine)
    size_of_test = len(test_to_determine)
    test_list = [1] * size_of_test
    x_train = np.concatenate((x_train, np.array(temp_list)[:, None]), axis=1)
    test_to_determine = np.concatenate((test_to_determine, np.array(test_list)[:, None]), axis=1)

    for curr_epoch in range(total_epoches):
        # Shuffling the lists of training set accordingly.
        # Inspiration: https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        shuffle_list = list(zip(x_train, y_train))
        np.random.shuffle(shuffle_list)

        sum_of_matches = 0
        for x, y in zip(x_train, y_train):
            maximal_prediction = np.argmax(np.dot(w, x))
            maximal_prediction = int(maximal_prediction)
            if maximal_prediction == y:  # If so, there's no need to update... matches sum is increased by 1
                sum_of_matches = sum_of_matches + 1
            else:
                w[y.astype(int)] = w[y.astype(int)] + np.array(x) * eta
                w[maximal_prediction] = w[maximal_prediction] - np.array(x) * eta

        print("epoch: {}, success rate: {}".format(curr_epoch, (sum_of_matches / size) * 100))
    for curr_test in test_to_determine:
        maximal_prediction = np.argmax(np.dot(w, curr_test))
        perceptron_output.append(maximal_prediction)

    return perceptron_output


if __name__ == "__main__":
    main()
