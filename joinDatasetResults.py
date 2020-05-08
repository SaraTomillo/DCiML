import csv
import pandas as pd
import numpy as np
import os
import friedman

headers = ["CV", "Kliep", "MKliep","KernelDensity", "MKernelDensity"]

def joinDatasetResults(dataset_name, percentage, seeds):
    distances = []
    for seed in seeds:
        directory = "./results-" + str(seed) + '/' + dataset_name
        filename = directory + "/" + dataset_name + "-" + str(percentage) + "-distances.csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data_array = np.array(data)
        for element in data_array:
            distances.append(element)


    directory = './results-final-' + str(percentage) + '/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + str(percentage) + "-distances.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(distances)
    return distances

def saveCSV(data, title, dataset_name, percentage):
    directory = './results-final-' + str(percentage) + '/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + str(percentage) + "-friedman.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(data)
    return

seeds = [2032, 2033, 2034, 2035, 2036]
#datasets = ["iris", "abalone", "ionosphere", "sonar"]

#datasets = ["cmc", "haberman", "SPECT", "wdbc"]
datasets = ["wine-quality-red", "bikes-day-casual", "bikes-day-registered", "bikes-day-total"]
percentage = 0.33

for dataset_name in datasets:
    distances = joinDatasetResults(dataset_name, percentage, seeds)
    ranking = friedman.friedman(distances, dataset_name, percentage)
    saveCSV(ranking, "friedman", dataset_name, percentage)