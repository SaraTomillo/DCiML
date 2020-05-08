import csv
import pandas as pd
import numpy as np
import os

headers = ["Eval", "CV", "Kliep", "MKliep","KernelDensity", "MKernelDensity"]

def readDataset(filename_train, filename_test):
    data_train = pd.read_csv(filename_train, sep=",", header=0)
    data_test = pd.read_csv(filename_test, sep=",", header=0)

    X_train = np.array(data_train.iloc[:, :-1])
    Y_train = np.array(data_train.iloc[:, -1])
    X_test = np.array(data_test.iloc[:, :-1])
    Y_test = np.array(data_test.iloc[:, -1])

    return X_train, Y_train, X_test, Y_test


def writeToCSV(data, dataset_name, execution_id, seed):
    directory = './results-' + str(seed) + '/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + execution_id + ".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows([data])
    return

def saveCSV(data, title, dataset_name, percentage, seed):
    directory = './results-' + str(seed) + '/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + str(percentage) + "-"+title+".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers[1:])
        writer.writerows(data)
    return
