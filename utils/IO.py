import csv
import pandas as pd
import numpy as np
import os


headers = []
headers_regression = ["Eval", "CV", "LR", "PLR", "CLR", "MLR", "KMM","PKMM", "CKMM", "MKMM", "KDE", "PKDE", "CKDE", "MKDE", "KLIEP", "PKLIEP", "CKLIEP", "MKLIEP"]
headers_classification = ["Eval", "CV", "LR", "PLR", "KMM","PKMM", "KDE", "PKDE", "KLIEP", "PKLIEP"]
headers_plankton = ["Eval", "CV", "LR", "PLR", "KDE", "PKDE"]

def readDataset(filename_train, filename_test):
    data_train = pd.read_csv(filename_train, sep=",", header=0)
    data_test = pd.read_csv(filename_test, sep=",", header=0)

    X_train = np.array(data_train.iloc[:, :-1])
    Y_train = np.array(data_train.iloc[:, -1])
    X_test = np.array(data_test.iloc[:, :-1])
    Y_test = np.array(data_test.iloc[:, -1])

    return X_train, Y_train, X_test, Y_test


def writeToCSV(data, problem_type, dataset_name, execution_id, seed):
    directory = './results/error_estimations/' + problem_type + '/'+ dataset_name
    filename = directory + "/" + dataset_name + "-" + str(seed)+ "-" + execution_id + ".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    headers = []
    if problem_type == "regression":
        headers = headers_regression
    elif problem_type == "classification":
        headers = headers_classification
    elif problem_type == "plankton":
        headers = headers_plankton


    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows([data])
    return

def saveCSV(data, title, problem_type, dataset_name, percentage, headers=headers):
    directory = './results/rankings/' + problem_type + '/' + dataset_name
    filename = directory + "/" + dataset_name + "-" +title + "-" + str(percentage) + ".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    if headers==headers:
        if problem_type == "regression":
            headers = headers_regression
        elif problem_type == "classification":
            headers = headers_classification
        elif problem_type == "plankton":
            headers = headers_plankton

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(data)
    return


def printResultsRegression(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M , ranks_triplets_P, ranks_triplets_M, CDs_triplets, wilcoxon):
    directory = './results/results-final'
    filename = directory + "/regression.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["FRIENDMAN ALL"])
        writer.writerow(["Dataset","CV", "LR", "PLR", "MLR", "KMM", "PKMM", "MKMM", "KDE", "PKDE", "MKDE", "KLIEP", "PKLIEP", "MKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_all[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_all:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''],[''],["FRIEDMAN WITHOUT M"]])
        writer.writerow(["Dataset", "CV", "LR", "PLR", "KMM", "PKMM", "KDE", "PKDE","KLIEP", "PKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_all_no_M[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_all_no_M:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''],[''],["FRIEDMAN BY METHODS"]])
        writer.writerow(["Dataset","CV", "LR", "PLR", "MLR", "CV", "KMM", "PKMM", "MKMM", "CV", "KDE", "PKDE", "MKDE", "CV", "KLIEP", "PKLIEP","MKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_methods[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_methods:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''], [''], ["FRIEDMAN TRIPLETS METHODS P"]])
        writer.writerow(
            ["Dataset", "CV", "LR", "PLR", "CV", "KMM", "PKMM", "CV", "KDE", "PKDE", "CV","KLIEP", "PKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_triplets_P[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_triplets:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''], [''], ["FRIEDMAN TRIPLETS METHODS M"]])
        writer.writerow(
            ["Dataset", "CV", "LR", "MLR", "CV","KMM", "MKMM", "CV","KDE", "MKDE","CV", "KLIEP", "MKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_triplets_M[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_triplets:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''], [''], ["WILCOXON"]])
        writer.writerow(["pair","wins 1st","wins 2nd","p-value"])
        pairs = ["LR-PLR", "LR-MLR", "KMM-PKMM", "KMM-MKMM", "KDE-PKDE", "KDE-MKDE", "KLIEP-PKLIEP", "KLIEP-MKLIEP"]
        for i in range(len(wilcoxon)):
            aux = []
            aux.append(pairs[i])
            for element in wilcoxon[i]:
                aux.append(element)
            writer.writerow(aux)
    return



def printResultsClassification(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, wilcoxon):
    directory = './results/results-final'
    filename = directory + "/classification.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["FRIENDMAN ALL"])
        writer.writerow(["Dataset","CV", "LR", "PLR", "KMM", "PKMM", "KDE", "PKDE", "KLIEP", "PKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_all[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_all:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''],[''],["FRIEDMAN BY METHODS"]])
        writer.writerow(["Dataset","CV", "LR", "PLR", "CV", "KMM", "PKMM", "CV", "KDE", "PKDE","CV", "KLIEP", "PKLIEP"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_methods[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_methods:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''], [''], ["WILCOXON"]])
        writer.writerow(["pair","wins 1st","wins 2nd","p-value"])
        pairs = ["LR-PLR", "KMM-PKMM", "KDE-PKDE", "KLIEP-PKLIEP"]
        for i in range(len(wilcoxon)):
            aux = []
            aux.append(pairs[i])
            for element in wilcoxon[i]:
                aux.append(element)
            writer.writerow(aux)
    return


def printResultsPlankton(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, wilcoxon):
    directory = './results/results-final'
    filename = directory + "/plankton.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["FRIENDMAN ALL"])
        writer.writerow(["Dataset","CV", "LR", "PLR", "KDE", "PKDE"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_all[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_all:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''],[''],["FRIEDMAN BY METHODS"]])
        writer.writerow(["Dataset","CV", "LR", "PLR", "CV", "KDE", "PKDE"])
        for i in range(len(datasets)):
            aux = []
            aux.append(datasets[i])
            for element in ranks_methods[i]:
                aux.append(element)
            writer.writerow(aux)
        writer.writerows([['']])
        writer.writerow(["alpha", 0.01, 0.05, 0.1])
        aux = []
        aux.append("CDs")
        for element in CDs_methods:
            aux.append(element)
        writer.writerow(aux)

        writer.writerows([[''], [''], ["WILCOXON"]])
        writer.writerow(["pair","wins 1st","wins 2nd","p-value"])
        pairs = ["LR-PLR", "KDE-PKDE"]
        for i in range(len(wilcoxon)):
            aux = []
            aux.append(pairs[i])
            for element in wilcoxon[i]:
                aux.append(element)
            writer.writerow(aux)
    return