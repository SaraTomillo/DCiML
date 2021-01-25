import numpy as np
import estimation_methods as estimation_methods


def partition(data, k):
    avg = len(data) / float(k)
    out = []
    last = 0.0

    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg

    return out


def ensemble_KMM_test(X_train, X_test, k=20, seed=2032):
    np.random.RandomState(seed)
    np.random.shuffle(X_test)
    partitions_test = partition(X_test, k)

    importances_partitions = []
    weights = []
    for test_partition in partitions_test:
        importance = estimation_methods.kmm(X_train, test_partition)
        weight = len(test_partition)/len(X_test)

        importances_partitions.append(importance)
        weights.append(weight)


    importances = np.concatenate(importances_partitions*weights)

    return importances

def ensemble_KMM_train(X_train, X_test, k=20, seed=2032):
    np.random.RandomState(seed)
    np.random.shuffle(X_train)
    partitions_train = partition(X_train, k)

    importances_partitions = []
    for train_partition in partitions_train:
        importance = estimation_methods.kmm(train_partition, X_test)
        importances_partitions.append(importance)

    importances = np.concatenate(importances_partitions)
    return importances


def ensemble_KMM_train_model(P_train, P_test):
    train = P_train

    train = []
    for element in P_train:
        train.append([element])
    train = np.array(train)

    test = []
    for element in P_test:
        test.append([element])
    test = np.array(test)

    importance = ensemble_KMM_train(train, test)
    return importance