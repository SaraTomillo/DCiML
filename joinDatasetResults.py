import csv
import pandas as pd
import numpy as np
import os
import friedman

headers = ["CV", "LR", "PLR" "KMM","PKMM", "KDE", "PKDE","KLIEP", "PKLIEP"]

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

    directory = './results-final-KMM/' + dataset_name
    filename = directory + "/" + dataset_name + "-" + str(seed) + "-" + str(percentage) + "-distances.csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(distances)
    return distances

def saveCSV(data, title, dataset_name, percentage, headers=headers):
    #directory = './results-final-' + str(percentage) + '/' + dataset_name
    #filename = directory + "/" + dataset_name + "-" +title + "-" + str(percentage) + ".csv"
    directory = './results-final/' + dataset_name
    filename = directory + "/" + dataset_name + "-" +title + "-" + str(percentage) + ".csv"

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
        writer.writerows(data)
    return

seeds = [2036]
datasets = ["iris", "sonar", "ionosphere", "cmc", "haberman",
            "transfusion", "wdbc","SPECT", "titanic","splice", "abalone",
            "computer-hardware", "wine-quality-red", "wine-quality-white",
            "auto-mpg", "autos", "residential-v9", "residential-v10","ticdata",
			"student-mat","student-por"]
percentage = 0.33

#datasets = ["plancton-2006", "plancton-2007", "plancton-2008", "plancton-2009", "plancton-2010", "plancton-2011", "plancton-2012", "plancton-2013"]
#seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]

for dataset_name in datasets:
    distances = joinDatasetResults(dataset_name, percentage, seeds)
    # KMM-MKMM
    distances_KMM_MKMM = []
    for element in distances:
        if not np.isnan(element[1]):
            distances_KMM_MKMM.append([element[1:3]])
    if len(distances_KMM_MKMM) >= 1:
        ranking = friedman.friedman_pairs(distances_KMM_MKMM, dataset_name, percentage)
        saveCSV(ranking, "friedman-KMM-MKMM", dataset_name, percentage, headers=["KMM", "MKMM"])
    # CV-KMM-MKMM
    distances_CV_KMM_MKMM = []
    for element in distances:
        if not np.isnan(element[1]):
            distances_CV_KMM_MKMM.append(element[0:3])
    if len(distances_CV_KMM_MKMM) >= 1:
        ranking = friedman.friedman(distances_CV_KMM_MKMM, dataset_name, percentage)
        saveCSV(ranking, "friedman-CV-KMM-MKMM", dataset_name, percentage, headers=["CV", "KMM", "MKMM"])

""" 
    # LogReg-MLogReg
    distances_LogReg_MLogReg = []
    for element in distances:
        if not np.isnan(element[1]):
            distances_LogReg_MLogReg.append([element[1:3]])
    if len(distances_LogReg_MLogReg)>= 1:
        ranking = friedman.friedman_pairs(distances_LogReg_MLogReg, dataset_name, percentage)
        saveCSV(ranking, "friedman-LogReg-MLogReg", dataset_name, percentage, headers=["LogReg", "MLogReg"])
    # CV-LogReg-MLogReg
    distances_CV_LogReg_MLogReg = []
    for element in distances:
        if not np.isnan(element[1]):
            distances_CV_LogReg_MLogReg.append(element[0:3])
    if len(distances_CV_LogReg_MLogReg) >= 1:
        ranking = friedman.friedman(distances_CV_LogReg_MLogReg, dataset_name, percentage)
        saveCSV(ranking, "friedman-CV-LogReg-MLogReg", dataset_name, percentage, headers=["CV", "LogReg", "MLogReg"])


    # KLiep-MKliep
    distances_Kliep_MKliep = []
    for element in distances:
        if not np.isnan(element[1]):
            distances_Kliep_MKliep.append([element[1:3]])
    if len(distances_Kliep_MKliep)>= 1:
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
        saveCSV(ranking, "friedman-CV-KLiep-MKliep", dataset_name, percentage, headers=["CV","Kliep", "MKliep"])

    # CV-KDE-MKDE
    distances_CV_KDE_MKDE = []
    for element in distances:
        distances_CV_KDE_MKDE.append([element[0], element[3], element[4]])
    ranking = friedman.friedman(distances_CV_KDE_MKDE, dataset_name, percentage)
    saveCSV(ranking, "friedman-CV-KDE-MKDE", dataset_name, percentage, headers=["CV", "KDE", "MKDE"])
"""

"""

data = pd.read_csv("results-final/plancton/plancton-2041-0.33-distances.csv", header = 0)
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