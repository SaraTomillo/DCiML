import csv
import pandas as pd
import os
from utils import friedman

#headers = ["CV", "Kliep", "MKliep","KernelDensity", "MKernelDensity"]
headers = ["CV", "LR", "PLR","KMM", "PKMM"]


def joinAllDistances(datasets, percentage, seeds):
    distances = []
    dataset_name = "plancton"
    seed = seeds[-1]
    for dataset in datasets:
        # directory = "./results-" + str(seed) + '/' + dataset_name
        # filename = directory + "/" + dataset_name + "-" + str(percentage) + "-distances.csv"
        directory = "./results-final/" + dataset
        filename = directory + "/" + dataset + "-" + str(seed) + "-" + str(percentage) + "-distances.csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)

    directory = './results-final/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + str(seed) + "-" + str(percentage) + "-distances.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(distances)
    return distances



def joinDatasetResults(dataset_name, percentage, seeds):
    distances = []
    for seed in seeds:
        #directory = "./results-" + str(seed) + '/' + dataset_name
        #filename = directory + "/" + dataset_name + "-" + str(percentage) + "-distances.csv"
        directory = "./results/" + dataset_name
        filename = directory + "/" + dataset_name + "-" + str(seed)+ "-" + str(percentage)+"-distances.csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)


    #directory = './results-final-' + str(percentage) + '/' + dataset_name
    #filename = directory + "/" + dataset_name + "-" + str(percentage) + "-distances.csv"

    directory = './results-final/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + str(seed) + "-" + str(percentage) + "-distances.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(distances)
    return distances

def saveCSV(data, title, dataset_name, percentage, headers=headers):
    directory = './results-final/' + dataset_name
    filename = directory + "/" + dataset_name + "-" +title + "-" + str(percentage) + ".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(data)
    return

def main():
    datasets = ["plancton-2006", "plancton-2007", "plancton-2008", "plancton-2009", "plancton-2010", "plancton-2011",
                "plancton-2012", "plancton-2013", "plancton-2006-2007", "plancton-2007-2008", "plancton-2008-2009",
                "plancton-2009-2010", "plancton-2010-2011", "plancton-2011-2012",
                "plancton-2012-2013"]

    seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
    percentage = 1.0

    for dataset_name in datasets:
        distances = joinDatasetResults(dataset_name, percentage, seeds)
        """
        # LR-PLR
        distances_Kliep_MKliep = []
        for element in distances:
            if not np.isnan(element[1]):
                distances_Kliep_MKliep.append([element[1:3]])
        if len(distances_Kliep_MKliep) >= 1:
            ranking = friedman.friedman_pairs(distances_Kliep_MKliep, dataset_name, percentage)
            saveCSV(ranking, "friedman-LR-PLR", dataset_name, percentage, headers=["LR", "PLR"])
        """
        # KMM-PKMM
        distances_KDE_MKDE = []
        for element in distances:
            distances_KDE_MKDE.append([element[3:]])
        ranking = friedman.friedman_pairs(distances_KDE_MKDE, dataset_name, percentage)
        saveCSV(ranking, "friedman-KDE-PKDE", dataset_name, percentage, headers=["KDE", "PKDE"])
        """
        # CV-LR-PLR
        distances_CV_Kliep_MKliep = []
        for element in distances:
            if not np.isnan(element[1]):
                distances_CV_Kliep_MKliep.append(element[0:3])
        if len(distances_CV_Kliep_MKliep) >= 1:
            ranking = friedman.friedman(distances_CV_Kliep_MKliep, dataset_name, percentage)
            saveCSV(ranking, "friedman-CV-LR-PLR", dataset_name, percentage, headers=["CV", "LR", "PLR"])
        """
        # CV-KMM-PKMM
        distances_CV_KDE_MKDE = []
        for element in distances:
            distances_CV_KDE_MKDE.append([element[0], element[3], element[4]])
        ranking = friedman.friedman(distances_CV_KDE_MKDE, dataset_name, percentage)
        saveCSV(ranking, "friedman-CV-KDE-PKDE", dataset_name, percentage, headers=["CV", "KDE", "PKDE"])

    joinAllDistances(datasets, percentage, seeds)
    data = pd.read_csv("results-final/plancton/plancton-2041-1.0-distances.csv", header = 0)
    dataset_name = "plancton"
    distances = data.values

    # LR-PLR
    distances_LR_PLR = []
    for element in distances:
        distances_LR_PLR.append([element[3:]])
    ranking = friedman.friedman_pairs(distances_LR_PLR, dataset_name, percentage)
    saveCSV(ranking, "friedman-LR-PLR", dataset_name, percentage, headers=["LR", "PLR"])
    # CV-LR-PLR
    distances_CV_KDE_MKDE = []
    for element in distances:
        distances_CV_KDE_MKDE.append([element[0], element[3], element[4]])
    ranking = friedman.friedman(distances_CV_KDE_MKDE, dataset_name, percentage)
    saveCSV(ranking, "friedman-CV-LR-PLR", dataset_name, percentage, headers=["CV", "LR", "PKMM"])
    # KMM-PKMM
    distances_KDE_MKDE = []
    for element in distances:
        distances_KDE_MKDE.append([element[3:]])
    ranking = friedman.friedman_pairs(distances_KDE_MKDE, dataset_name, percentage)
    saveCSV(ranking, "friedman-KMM-PKMM", dataset_name, percentage, headers=["KMM", "PKMM"])
    # CV-KMM-PKMM
    distances_CV_KDE_MKDE = []
    for element in distances:
        distances_CV_KDE_MKDE.append([element[0], element[3], element[4]])
    ranking = friedman.friedman(distances_CV_KDE_MKDE, dataset_name, percentage)
    saveCSV(ranking, "friedman-CV-KMM-PKMM", dataset_name, percentage, headers=["CV", "KMM", "PKMM"])


"""
        # KLiep-MKliep
        distances_Kliep_MKliep = []
        for element in distances:
            if not np.isnan(element[1]):
                distances_Kliep_MKliep.append([element[1:3]])
        if len(distances_Kliep_MKliep) >= 1:
            ranking = friedman.friedman_pairs(distances_Kliep_MKliep, dataset_name, percentage)
            saveCSV(ranking, "friedman-Kliep-MKliep", dataset_name, percentage, headers=["Kliep", "MKliep"])

        # KDE-MKDE
        distances_KDE_MKDE = []
        for element in distances:
            distances_KDE_MKDE.append([element[3:]])
        ranking = friedman.friedman_pairs(distances_KDE_MKDE, dataset_name, percentage)
        saveCSV(ranking, "friedman-KDE-MKDE", dataset_name, percentage, headers=["KDE", "MKDE"])

        # CV-KLiep-MKliep
        distances_CV_Kliep_MKliep = []
        for element in distances:
            if not np.isnan(element[1]):
                distances_CV_Kliep_MKliep.append(element[0:3])
        if len(distances_CV_Kliep_MKliep) >= 1:
            ranking = friedman.friedman(distances_CV_Kliep_MKliep, dataset_name, percentage)
            saveCSV(ranking, "friedman-CV-KLiep-MKliep", dataset_name, percentage, headers=["CV", "Kliep", "MKliep"])

        # CV-KDE-MKDE
        distances_CV_KDE_MKDE = []
        for element in distances:
            distances_CV_KDE_MKDE.append([element[0], element[3], element[4]])
        ranking = friedman.friedman(distances_CV_KDE_MKDE, dataset_name, percentage)
        saveCSV(ranking, "friedman-CV-KDE-MKDE", dataset_name, percentage, headers=["CV", "KDE", "MKDE"])

    joinAllDistances(datasets, percentage, seeds)
    data = pd.read_csv("results-final/plancton/plancton-2041-1.0-distances.csv", header = 0)
    dataset_name = "plancton"
    distances = data.values
    distances_KDE_MKDE = []
    for element in distances:
        distances_KDE_MKDE.append([element[3:]])
    ranking = friedman.friedman_pairs(distances_KDE_MKDE, dataset_name, percentage)
    saveCSV(ranking, "friedman-KDE-MKDE", dataset_name, percentage, headers=["KDE", "MKDE"])
    # CV-KDE-MKDE
    distances_CV_KDE_MKDE = []
    for element in distances:
        distances_CV_KDE_MKDE.append([element[0], element[3], element[4]])
    ranking = friedman.friedman(distances_CV_KDE_MKDE, dataset_name, percentage)
    saveCSV(ranking, "friedman-CV-KDE-MKDE", dataset_name, percentage, headers=["CV", "KDE", "MKDE"])
"""



if __name__ == "__main__":
    main()
