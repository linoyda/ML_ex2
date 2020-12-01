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
    alg_labels = knn_algorithm_implementation(test_to_determine, best_k, training_arr, label_arr)
    print(alg_labels)


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


if __name__ == "__main__":
    main()
