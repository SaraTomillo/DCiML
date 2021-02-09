import pandas as pd
from utils import wilcoxon, friedman
from utils.IO import saveCSV

def friedman_methods(datasets, problem, percentage):
    ranksMethods = []
    CDs = []
    aux = []
    for dataset_name in datasets:
        distances =[]
        directory = "./results/rankings/" + problem + "/" + dataset_name
        filename = directory + "/" + dataset_name + "-distances-" + str(percentage) + ".csv"
        print(dataset_name)
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)

        avgRanks = []

        # CV-LR-PLR
        distances_LR_PLR_BLR = []
        for element in distances:
            distances_LR_PLR_BLR.append([element[1], element[2], element[3]])
        ranking, avgRank, CDs = friedman.friedman(distances_LR_PLR_BLR)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-LR-PLR-BLR", problem, dataset_name, percentage, headers=["CV", "LR", "PLR", "BLR"])
        # CV-KMM-PKMM
        distances_KMM_PKMM_BKMM = []
        for element in distances:
            distances_KMM_PKMM_BKMM.append([element[4], element[5], element[6]])
        ranking, avgRank, CDs = friedman.friedman(distances_KMM_PKMM_BKMM)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KMM-PKMM-BKMM",problem, dataset_name, percentage, headers=["CV", "KMM", "PKMM", "BKMM"])
        # CV-KDE-PKDE-BKDE
        distances_KDE_PKDE_BKDE = []
        for element in distances:
            distances_KDE_PKDE_BKDE.append([element[7], element[8], element[9]])
        ranking, avgRank, CDs = friedman.friedman(distances_KDE_PKDE_BKDE)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KDE-PKDE-BKDE",problem, dataset_name, percentage, headers=["CV", "KDE", "PKDE", "BKDE"])
        # CV-KLIEP-PKLIEP-BKLIEP
        distances_KLIEP_PKLIEP_BKLIEP = []
        for element in distances:
            distances_KLIEP_PKLIEP_BKLIEP.append([element[10], element[11], element[12]])
        ranking, avgRank, CDs = friedman.friedman(distances_KLIEP_PKLIEP_BKLIEP)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KLIEP-PKLIEP-BKLIEP", problem, dataset_name, percentage, headers=["CV", "KLIEP", "PKLIEP", "BKLIEP"])

        # CV-KMM-PKMM-BKMM-ENS
        distances_KMM_PKMM_BKMM_ENS = []
        for element in distances:
            distances_KMM_PKMM_BKMM_ENS.append([element[13], element[14], element[15]])
        ranking, avgRank, CDs = friedman.friedman(distances_KMM_PKMM_BKMM_ENS)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KMM-PKMM-BKMM-ENS", problem, dataset_name, percentage,
                headers=["CV", "KMM-ENS", "PKMM-ENS", "BKMM-ENS"])

        ranks = []
        for element in avgRanks:
            for each in element:
                ranks.append(each)
        ranksMethods.append(ranks)

        for element in ranking:
            if len(element) == 0:
                break
            aux.append(element)

    saveCSV(ranksMethods, "friedman-methods", problem, problem, percentage,
                headers=["CV", "LR", "PLR", "BLR", "CV", "KMM", "PKMM", "BKMM", "CV", "KDE", "PKDE", "BKDE", "CV", "KLIEP", "PKLIEP", "BKLIEP", "CV", "KMM-ENS", "PKMM-ENS", "BKMM-ENS"])
    return ranksMethods, CDs, aux


def friedman_all(datasets, problem, percentage):
    ranksAll = []
    CDs = []
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
            distances_ALL.append([element[0], element[1], element[2], element[3], element[4], element[5], element[6], element[7], element[8],
                                  element[9], element[10], element[11], element[12], element[13], element[14], element[15]])
        ranking, avgRank, CDs = friedman.friedman(distances_ALL)
        saveCSV(ranking, "friedman-ALL",problem, dataset_name, percentage,
                headers=["CV", "LR", "PLR","BLR", "KMM", "PKMM", "BKMM", "KDE", "PKDE", "BKDE", "KLIEP", "PKLIEP", "BKLIEP", "KMM-ENS", "PKMM-ENS", "BKMM-ENS"])

        ranksAll.append(avgRank)
    saveCSV(ranksAll, "friedman-all", problem, problem, percentage, headers=["CV", "LR", "PLR", "BLR", "KMM", "PKMM","BKMM", "KDE", "PKDE","BKDE", "KLIEP", "PKLIEP", "BKLIEP", "KMM-ENS", "PKMM-ENS", "BKMM-ENS" ])
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
    #saveCSV([[wins, loses, p]], "wilcoxon-LR-PLR", problem, problem, percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KMM_PKMM)
    #saveCSV([[wins, loses, p]], "wilcoxon-KMM-PKMM", problem, problem, percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KDE_PKDE)
    #saveCSV([[wins, loses, p]], "wilcoxon-KDE-PKDE", problem, problem, percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    wins, loses, p = wilcoxon.wilcoxon(distances_KLIEP_PKLIEP)
    #saveCSV([[wins, loses, p]], "wilcoxon-KLIEP-PKLIEP", problem, problem, percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])

    saveCSV(all_rank, "wilcoxon-all", problem, problem, percentage, headers=["wins", "loses", "p-value"])
    return all_rank
