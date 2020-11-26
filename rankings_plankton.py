import pandas as pd
from utils import wilcoxon, friedman
from utils.IO import saveCSV

def friedman_methods(datasets, problem, percentage):
    ranksMethods = []
    for dataset_name in datasets:
        distances =[]
        directory = "./results/rankings/" + problem + "/" + dataset_name
        filename = directory + "/" + dataset_name + "-distances-" + str(percentage) + ".csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)

        avgRanks = []

        # CV-LR-PLR
        distances_CV_LR_PLR_MLR = []
        for element in distances:
            distances_CV_LR_PLR_MLR.append([element[0], element[1], element[2], element[3]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_LR_PLR_MLR)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-LR-PLR","plankton", dataset_name, percentage, headers=["CV", "LR", "PLR", "BLR"])

        # CV-KDE-PKDE
        distances_CV_KDE_PKDE_MKDE = []
        for element in distances:
            distances_CV_KDE_PKDE_MKDE.append([element[0], element[4], element[5], element[6]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KDE_PKDE_MKDE)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KDE-PKDE","plankton", dataset_name, percentage, headers=["CV", "KDE", "PKDE", "BKDE"])
        ranks = []
        for element in avgRanks:
            for each in element:
                ranks.append(each)
        ranksMethods.append(ranks)

    saveCSV(ranksMethods, "friedman-methods", "plankton", "plankton", percentage, headers=["CV", "LR", "PLR", "BLR", "CV", "KDE", "PKDE", "BKDE"])
    return ranksMethods, CDs


def friedman_all(datasets, problem, percentage):
    avgRanks = []
    for dataset_name in datasets:
        distances =[]
        directory = "./results/rankings/" + problem + "/" + dataset_name
        filename = directory + "/" + dataset_name + "-distances-" + str(percentage) + ".csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)

        distances_ALL = []
        for element in distances:
            distances_ALL.append(element[:])
        ranking, avgRank, CDs = friedman.friedman(distances_ALL)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-ALL","plankton", dataset_name, percentage,
                headers=["CV", "LR", "PLR", "KDE", "PKDE"])

    saveCSV(avgRanks, "friedman-all", "plankton", "plankton", percentage, headers=["CV", "LR", "PLR", "BLR", "KDE", "PKDE", "BKDE"])
    return avgRanks, CDs


def wilcoxon_rank(datasets, problem, percentage):
    distances_LR_PLR = []
    distances_LR_BLR = []
    distances_PLR_BLR = []
    distances_KDE_PKDE = []
    distances_KDE_BKDE = []
    distances_PKDE_BKDE = []
    for dataset_name in datasets:
        distances =[]
        directory = "./results/rankings/" + problem + "/" + dataset_name
        filename = directory + "/" + dataset_name + "-distances-" + str(percentage) + ".csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values

        for element in data:
            distances.append(element)
        # LR-PLR
        for element in distances:
            distances_LR_PLR.append([element[1], element[2]])
        # LR-BLR
        for element in distances:
            distances_LR_BLR.append([element[1], element[3]])
        # PLR-BLR
        for element in distances:
            distances_PLR_BLR.append([element[2], element[3]])

        # KDE-PKDE
        for element in distances:
            distances_KDE_PKDE.append([element[4], element[5]])
        # KDE-BKDE
        for element in distances:
            distances_KDE_BKDE.append([element[4], element[6]])
        # PKDE-BKDE
        for element in distances:
            distances_PKDE_BKDE.append([element[5], element[6]])

    all_rank = []
    wins, loses, p = wilcoxon.wilcoxon(distances_LR_PLR)
    #saveCSV([[wins, loses, p]], "wilcoxon-LR-PLR", "plankton", "plankton", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_LR_BLR)
    # saveCSV([[wins, loses, p]], "wilcoxon-LR-BLR", "plankton", "plankton", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_PLR_BLR)
    # saveCSV([[wins, loses, p]], "wilcoxon-PLR-BLR", "plankton", "plankton", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KDE_PKDE)
    #saveCSV([[wins, loses, p]], "wilcoxon-KDE-PKDE", "plankton", "plankton", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KDE_BKDE)
    # saveCSV([[wins, loses, p]], "wilcoxon-KDE-BKDE", "plankton", "plankton", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_PKDE_BKDE)
    # saveCSV([[wins, loses, p]], "wilcoxon-PKDE-BKDE", "plankton", "plankton", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    saveCSV(all_rank, "wilcoxon-ALL", "plankton","plankton", percentage, headers=["wins", "loses", "p-value"])
    return all_rank