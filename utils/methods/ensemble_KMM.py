import numpy as np
import estimation_methods as estimation_methods
from utils.methods import KMM_optimized


def partition(data, k):
    avg = len(data) / float(k)
    out = []
    last = 0.0

    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg

    return out


def ensemble_KMM_test(X_train, X_test, k):
    random = np.RandomState(2032)

    shuffled_test = np.shuffle(X_test, random)
    partitions_test = partition(shuffled_test, k)

    importances_partitions = []
    weights = []
    for test_partition in partitions_test:
        importance = estimation_methods.kmm(X_train, test_partition)
        #importance = KMM_optimized.kmm(X_train, test_partition)
        weight = len(test_partition)/len(X_test)

        importances_partitions.append(importance)
        weights.append(weight)


    importances = np.sum(importances_partitions*weights)

    return importances


def ensemble_KMM_train(X_train, X_test):
    random = np.RandomState(2032)

    shuffled_train = np.shuffle(X_train, random)
    partitions_train = partition(shuffled_train, k)

    importances_partitions = []
    for train_partition in partitions_train:
        importance = estimation_methods.kmm(train_partition, X_test)
        #importance = KMM_optimized.kmm(train_partition, X_test)
        importances_partitions.append(importance)

    importances = np.sum(importances_partitions)

    return importances



import utils.IO as IO
filename_train = "datasets/classification/iris/datasets-2032/iris-train-0.33.csv"
filename_test = "datasets/classification/iris/datasets-2032/iris-test-0.33-0.csv"
X_train, Y_train, X_test, Y_test = IO.readDataset(filename_train, filename_test)

print(estimation_methods.kmm(X_train, X_test))
print()
print(ensemble_KMM_test(X_train, X_test, 2))
print()
print(ensemble_KMM_train(X_train, X_test, 2))