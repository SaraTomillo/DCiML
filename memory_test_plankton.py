import csv
import os
import traceback

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

import estimation_methods
from utils import IO

import tracemalloc
from utils.utils import minmax, reduce, npShuffle, display_top
import time
import warnings
warnings.filterwarnings("ignore")

def error(predictions, y):
    return np.sign(np.abs(predictions-y))


bandwith = 1
kernel = 'linear'

def classification_KMM(X_train, Y_train, X_test, Y_test, seed):
    memory_used = 0
    elapsed_time = 0
    tracemalloc.start()

    model = SVC(kernel='linear', C=1, coef0=0)

    minmax_X = minmax(X_train)
    X_train = reduce(X_train, minmax_X)
    X_test = reduce(X_test, minmax_X)

    model.fit(X_train, Y_train)
    Pred_Eval = model.predict(X_test)

    eval = error(Pred_Eval, Y_test)

    X_train=npShuffle(X_train,seed)
    Y_train=npShuffle(Y_train,seed)

    CV = cross_val_predict(model, X_train, Y_train, cv=10)#, method='decision_function')
    CVError = error(CV, Y_train)

    importanceModel = SVC(kernel='linear', C=1, class_weight='balanced', coef0=0)
    importanceModel.fit(X_train, Y_train)

    P_train = importanceModel.decision_function(X_train)
    P_test = importanceModel.decision_function(X_test)

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    tracemalloc.start()
    start_time = time.time()
    try:
        importances_KMM = estimation_methods.kmm(Y_train, Y_test)
        KMM_err = CVError * importances_KMM
    except Exception:
        print("Error KMM")
        KMM_err = []
        traceback.print_exc()

    elapsed_time = time.time() - start_time
    snapshot = tracemalloc.take_snapshot()
    memory_used = display_top(snapshot)

    return memory_used, elapsed_time



def classification_KLIEP(X_train, Y_train, X_test, Y_test, seed):
    memory_used = 0
    elapsed_time = 0

    model = SVC(kernel='linear', C=1, coef0=0)

    minmax_X = minmax(X_train)
    X_train = reduce(X_train, minmax_X)
    X_test = reduce(X_test, minmax_X)

    model.fit(X_train, Y_train)
    Pred_Eval = model.predict(X_test)

    eval = error(Pred_Eval, Y_test)

    X_train=npShuffle(X_train,seed)
    Y_train=npShuffle(Y_train,seed)

    CV = cross_val_predict(model, X_train, Y_train, cv=10)#, method='decision_function')
    CVError = error(CV, Y_train)

    importanceModel = SVC(kernel='linear', C=1, class_weight='balanced', coef0=0)
    importanceModel.fit(X_train, Y_train)

    P_train = importanceModel.decision_function(X_train)
    P_test = importanceModel.decision_function(X_test)

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    tracemalloc.start()
    start_time = time.time()
    try:
        importances_KLIEP = estimation_methods.kliep(Y_train, Y_test)
        KLIEP_err = CVError * importances_KLIEP
    except Exception:
        print("Error KLIEP")
        KLIEP_err = []
        traceback.print_exc()

    elapsed_time = time.time() - start_time
    snapshot = tracemalloc.take_snapshot()
    memory_used = display_top(snapshot)

    return memory_used, elapsed_time



def test_memory_KMM(filename_train, filename_test, percentages):
    seed = 2032
    X_train, Y_train, X_test, Y_test = IO.readDataset(filename_train, filename_test)

    memory = []
    times = []
    for percentage in percentages:
        print("KMM at %s percentage of the dataset" % percentage)
        length_train = len(X_train)
        samples_train = int((length_train * percentage)/100)
        X_train_slice = X_train[:samples_train]
        Y_train_slice = Y_train[:samples_train]

        length_test = len(X_test)
        samples_test = int((length_test * percentage)/100)
        X_test_slice = X_test[:samples_test]
        Y_test_slice = Y_test[:samples_test]

        memory_used, elapsed_time = classification_KMM(X_train_slice, Y_train_slice, X_test_slice, Y_test_slice, seed)
        memory.append(memory_used)
        times.append(elapsed_time)

    return memory, times



def test_memory_KLIEP(filename_train, filename_test, percentages):
    seed = 2032
    X_train, Y_train, X_test, Y_test = IO.readDataset(filename_train, filename_test)

    memory = []
    times = []
    for percentage in percentages:
        print("KMM at %s percentage of the dataset" % percentage)
        length_train = len(X_train)
        samples_train = int((length_train * percentage)/100)
        X_train_slice = X_train[:samples_train]
        Y_train_slice = Y_train[:samples_train]

        length_test = len(X_test)
        samples_test = int((length_test * percentage)/100)
        X_test_slice = X_test[:samples_test]
        Y_test_slice = Y_test[:samples_test]

        memory_used, elapsed_time = classification_KLIEP(X_train_slice, Y_train_slice, X_test_slice, Y_test_slice, seed)
        memory.append(memory_used)
        times.append(elapsed_time)

    return memory, times



def writeToCSV(data, dataset_name, method_name):
    directory = './results/memory-time-test/'
    filename = directory + "/" + dataset_name + "-" + method_name + ".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Percentage", "Memory (KiB)", "Time (s)"])
        writer.writerows(data)
    return

def main():
    percentages =  range(10, 110, 10)
    filename_train = "./datasets/plankton/plankton-2007-1.csv"
    filename_test = "./datasets/plankton/plankton-2007-2.csv"

    memory, times = test_memory_KMM(filename_train, filename_test, percentages)

    data = []
    for i in range(len(percentages)):
        row = []
        row.append(percentages[i])
        row.append(memory[i])
        row.append(times[i])
        data.append(row)
    writeToCSV(data, "plankton-2007", "KMM")

    memory, times = test_memory_KLIEP(filename_train, filename_test, percentages)

    data = []
    for i in range(len(percentages)):
        row = []
        row.append(percentages[i])
        row.append(memory[i])
        row.append(times[i])
        data.append(row)
    writeToCSV(data, "plankton-2007", "KLIEP")


if __name__ == "__main__":
    main()
