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
        distances_CV_LR_PLR_BLR = []
        for element in distances:
            distances_CV_LR_PLR_BLR.append([element[0], element[1], element[2], element[3]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_LR_PLR_BLR)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-LR-PLR-BLR", "classification", dataset_name, percentage, headers=["CV", "LR", "PLR", "BLR"])
        # CV-KMM-PKMM
        distances_CV_KMM_PKMM_BKMM = []
        for element in distances:
            distances_CV_KMM_PKMM_BKMM.append([element[0], element[4], element[5], element[6]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KMM_PKMM_BKMM)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KMM-PKMM-BKMM","classification", dataset_name, percentage, headers=["CV", "KMM", "PKMM", "BKMM"])
        # CV-KDE-PKDE
        distances_CV_KDE_PKDE_BKDE = []
        for element in distances:
            distances_CV_KDE_PKDE_BKDE.append([element[0], element[7], element[8], element[9]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KDE_PKDE_BKDE)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KDE-PKDE","classification", dataset_name, percentage, headers=["CV", "KDE", "PKDE", "BKDE"])
        # CV-KLIEP-PKLIEP
        distances_CV_KLIEP_PKLIEP_BKLIEP = []
        for element in distances:
            distances_CV_KLIEP_PKLIEP_BKLIEP.append([element[0], element[10], element[11], element[12]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KLIEP_PKLIEP_BKLIEP)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KLIEP-PKLIEP","classification", dataset_name, percentage, headers=["CV", "KLIEP", "PKLIEP", "BKLIEP"])

        ranks = []
        for element in avgRanks:
            for each in element:
                ranks.append(each)
        ranksMethods.append(ranks)

    saveCSV(ranksMethods, "friedman-methods", "classification", "classification", percentage,
                headers=["CV", "LR", "PLR", "BLR", "CV", "KMM", "PKMM", "BKMM", "CV", "KDE", "PKDE", "BKDE", "CV", "KLIEP", "PKLIEP", "BKLIEP"])
    return ranksMethods, CDs


def friedman_all(datasets, problem, percentage):
    ranksAll = []
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
            distances_ALL.append([element[0], element[1], element[2], element[3], element[4], element[5], element[6], element[7], element[8]])
        ranking, avgRank, CDs = friedman.friedman(distances_ALL)
        saveCSV(ranking, "friedman-ALL","classification", dataset_name, percentage,
                headers=["CV", "LR", "PLR", "KMM", "PKMM", "KDE", "PKDE", "KLIEP", "PKLIEP"])

        ranksAll.append(avgRank)
    saveCSV(ranksAll, "friedman-all", "classification", "classification", percentage, headers=["CV", "LR", "PLR", "KMM", "PKMM", "KDE", "PKDE", "KLIEP", "PKLIEP"])
    return ranksAll, CDs


def wilcoxon_rank(datasets, problem, percentage):
    distances_LR_PLR = []
    distances_KMM_PKMM = []
    distances_KDE_PKDE = []
    distances_KLIEP_PKLIEP = []
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

        # KMM-PKMM
        for element in distances:
            distances_KMM_PKMM.append([element[3], element[4]])

        # KDE-PKDE
        for element in distances:
            distances_KDE_PKDE.append([element[5], element[6]])

        # KLIEP-PKLIEP
        for element in distances:
            distances_KLIEP_PKLIEP.append([element[7], element[8]])

    all_rank = []
    wins, loses, p = wilcoxon.wilcoxon(distances_LR_PLR)
    #saveCSV([[wins, loses, p]], "wilcoxon-LR-PLR", "classification", "classification", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KMM_PKMM)
    #saveCSV([[wins, loses, p]], "wilcoxon-KMM-PKMM", "classification", "classification", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KDE_PKDE)
    #saveCSV([[wins, loses, p]], "wilcoxon-KDE-PKDE", "classification", "classification", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KLIEP_PKLIEP)
    #saveCSV([[wins, loses, p]], "wilcoxon-KLIEP-PKLIEP", "classification", "classification", percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    saveCSV(all_rank, "wilcoxon-all", "classification", "classification", percentage, headers=["wins", "loses", "p-value"])
    return all_rank
