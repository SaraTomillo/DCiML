import pandas as pd
from utils import wilcoxon, friedman
from utils.IO import saveCSV

def friedman_triplets(datasets, problem, percentage):
    ranksP = []
    ranksM = []
    for dataset_name in datasets:
        distances =[]
        directory = "./results/rankings/" + problem +"/" + dataset_name
        filename = directory + "/" + dataset_name + "-distances-" + str(percentage) + ".csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)
        avgRanksP = []
        avgRanksM = []
        # CV-LR-PLR
        distances_CV_LR_PLR = []
        for element in distances:
            distances_CV_LR_PLR.append([element[0], element[1], element[2]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_LR_PLR)
        avgRanksP.append(avgRank)
        saveCSV(ranking, "friedman-CV-LR-PLR","regression", dataset_name, percentage, headers=["CV", "LR", "PLR"])
        # CV-LR-MLR
        distances_CV_LR_MLR = []
        for element in distances:
            distances_CV_LR_MLR.append([element[0], element[1], element[4]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_LR_MLR)
        avgRanksM.append(avgRank)
        saveCSV(ranking, "friedman-CV-LR-MLR","regression", dataset_name, percentage, headers=["CV", "LR", "MLR"])
        # CV-KMM-PKMM
        distances_CV_KMM_PKMM = []
        for element in distances:
            distances_CV_KMM_PKMM.append([element[0], element[5], element[6]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KMM_PKMM)
        avgRanksP.append(avgRank)
        saveCSV(ranking, "friedman-CV-KMM-PKMM","regression", dataset_name, percentage, headers=["CV", "KMM", "PKMM"])
        # CV-KMM-MKMM
        distances_CV_KMM_MKMM = []
        for element in distances:
            distances_CV_KMM_MKMM.append([element[0], element[5], element[8]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KMM_MKMM)
        avgRanksM.append(avgRank)
        saveCSV(ranking, "friedman-CV-KMM-MKMM","regression", dataset_name, percentage, headers=["CV", "KMM", "MKMM"])
        # CV-KDE-PKDE
        distances_CV_KDE_PKDE = []
        for element in distances:
            distances_CV_KDE_PKDE.append([element[0], element[9], element[10]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KDE_PKDE)
        avgRanksP.append(avgRank)
        saveCSV(ranking, "friedman-CV-KDE-PKDE","regression", dataset_name, percentage, headers=["CV", "KDE", "PKDE"])
        # CV-KDE-MKDE
        distances_CV_KDE_MKDE = []
        for element in distances:
            distances_CV_KDE_MKDE.append([element[0], element[9], element[12]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KDE_MKDE)
        avgRanksM.append(avgRank)
        saveCSV(ranking, "friedman-CV-KDE-PKDE-MKDE","regression", dataset_name, percentage, headers=["CV", "KDE", "MKDE"])
        # CV-KLIEP-PKLIEP
        distances_CV_KLIEP_PKLIEP = []
        for element in distances:
            distances_CV_KLIEP_PKLIEP.append([element[0], element[13], element[14]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KLIEP_PKLIEP)
        avgRanksP.append(avgRank)
        saveCSV(ranking, "friedman-CV-KLIEP-PKLIEP","regression", dataset_name, percentage, headers=["CV", "KLIEP", "PKLIEP"])
        # CV-KLIEPMKLIEP
        distances_CV_KLIEP_MKLIEP = []
        for element in distances:
            distances_CV_KLIEP_MKLIEP.append([element[0], element[13], element[16]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KLIEP_MKLIEP)
        avgRanksM.append(avgRank)
        saveCSV(ranking, "friedman-CV-KLIEP-MKLIEP","regression", dataset_name, percentage, headers=["CV", "KLIEP", "MKLIEP"])

        ranks = []
        for element in avgRanksP:
            for each in element:
                ranks.append(each)
        ranksP.append(ranks)
        ranks = []
        for element in avgRanksM:
            for each in element:
                ranks.append(each)
        ranksM.append(ranks)

    saveCSV(ranksP, "friedman-triplets-P", "regression", "regression", percentage, headers=["CV", "LR", "PLR", "CV", "KMM", "PKMM", "CV", "KDE", "PKDE", "CV","KLIEP", "PKLIEP"])
    saveCSV(ranksM, "friedman-triplets-M", "regression", "regression", percentage,  headers=["CV", "LR", "MLR", "CV","KMM", "MKMM", "CV","KDE", "MKDE","CV", "KLIEP", "MKLIEP"])

    return ranksP, ranksM, CDs

def friedman_methods(datasets, problem, percentage):
    ranksMethods= []
    for dataset_name in datasets:
        distances =[]
        directory = "./results/rankings/" + problem + "/" + dataset_name
        filename = directory + "/" + dataset_name + "-distances-" + str(percentage) + ".csv"
        data = pd.read_csv(filename, sep=",", header=0)
        data = data.values
        for element in data:
            distances.append(element)

        avgRanks = []

        # CV-LR-PLR-MLR
        distances_CV_LR_PLR_MLR = []
        for element in distances:
            distances_CV_LR_PLR_MLR.append([element[0], element[1], element[2], element[4]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_LR_PLR_MLR)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-LR-PLR-MLR","regression", dataset_name, percentage, headers=["CV", "LR", "PLR", "MLR"])
        # CV-KMM-PKMM-MKMM
        distances_CV_KMM_PKMM_MKMM = []
        for element in distances:
            distances_CV_KMM_PKMM_MKMM.append([element[0], element[5], element[6], element[8]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KMM_PKMM_MKMM)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KMM-PKMM-MKMM","regression", dataset_name, percentage, headers=["CV", "KMM", "PKMM", "MKMM"])
        # CV-KDE-PKDE-MKDE
        distances_CV_KDE_PKDE_MKDE = []
        for element in distances:
            distances_CV_KDE_PKDE_MKDE.append([element[0], element[9], element[10], element[12]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KDE_PKDE_MKDE)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KDE-PKDE-MKDE","regression", dataset_name, percentage, headers=["CV", "KDE", "PKDE", "MKDE"])
        # CV-KLIEP-PKLIEP-MKLIEP
        distances_CV_KLIEP_PKLIEP_MKLIEP = []
        for element in distances:
            distances_CV_KLIEP_PKLIEP_MKLIEP.append([element[0], element[13], element[14], element[16]])
        ranking, avgRank, CDs = friedman.friedman(distances_CV_KLIEP_PKLIEP_MKLIEP)
        avgRanks.append(avgRank)
        saveCSV(ranking, "friedman-CV-KLIEP-PKLIEP-MKLIEP","regression", dataset_name, percentage, headers=["CV", "KLIEP", "PKLIEP", "MKLIEP"])

        ranks = []
        for element in avgRanks:
            for each in element:
                ranks.append(each)
        ranksMethods.append(ranks)

        saveCSV(ranksMethods, "friedman-methods", "regression", "regression", percentage,
                headers=["CV", "LR", "PLR", "MLR", "CV", "KMM", "PKMM", "MKMM", "CV", "KDE", "PKDE", "MKDE", "CV", "KLIEP", "PKLIEP","MKLIEP"])
        return ranksMethods, CDs


def friedman_all(datasets, problem, percentage):
    ranksAll = []
    ranksAllNoM = []
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
            distances_ALL.append([element[0], element[1], element[2], element[4], element[5], element[6], element[8],
                                  element[9], element[10], element[12], element[13], element[14], element[16]])
        ranking, avgRank, CDs = friedman.friedman(distances_ALL)
        ranksAll.append(avgRank)
        saveCSV(ranking, "friedman-ALL","regression", dataset_name, percentage,
                headers=["CV", "LR", "PLR", "MLR", "KMM", "PKMM", "MKMM", "KDE", "PKDE", "MKDE", "KLIEP", "PKLIEP", "MKLIEP"])

        distances_ALL_No_M = []
        for element in distances:
            distances_ALL_No_M.append([element[0], element[1], element[2], element[5], element[6],
                                  element[9], element[10], element[13], element[14]])
        ranking, avgRank, CDs_no_M = friedman.friedman(distances_ALL_No_M)
        ranksAllNoM.append(avgRank)
        saveCSV(ranking, "friedman-ALL-no-M","regression", dataset_name, percentage,
                headers=["CV", "LR", "PLR", "KMM", "PKMM", "KDE", "PKDE", "KLIEP", "PKLIEP"])

    saveCSV(ranksAll, "friedman-all", "regression", "regression",  percentage, headers=["CV", "LR", "PLR", "MLR", "KMM", "PKMM", "MKMM", "KDE", "PKDE", "MKDE", "KLIEP", "PKLIEP", "MKLIEP"])
    saveCSV(ranksAllNoM, "friedman-all-no-M", "regression", "regression",  percentage, headers=["CV", "LR", "PLR", "KMM", "PKMM", "KDE", "PKDE", "KLIEP", "PKLIEP"])

    return ranksAll, CDs, ranksAllNoM, CDs_no_M


def wilcoxon_rank(datasets, problem, percentage):
    distances_LR_PLR = []
    distances_LR_MLR = []
    distances_KMM_PKMM = []
    distances_KMM_MKMM = []
    distances_KDE_PKDE = []
    distances_KDE_MKDE = []
    distances_KLIEP_PKLIEP = []
    distances_KLIEP_MKLIEP = []
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

        # LR-MLR
        for element in distances:
            distances_LR_MLR.append([element[1], element[4]])

        # KMM-PKMM
        for element in distances:
            distances_KMM_PKMM.append([element[5], element[6]])

        # KMM-MKMM
        for element in distances:
            distances_KMM_MKMM.append([element[5], element[8]])

        # KDE-PKDE
        for element in distances:
            distances_KDE_PKDE.append([element[9], element[10]])

        # KDE-MKDE
        for element in distances:
            distances_KDE_MKDE.append([element[9], element[12]])

        # KLIEP-PKLIEP
        for element in distances:
            distances_KLIEP_PKLIEP.append([element[13], element[14]])

        # KLIEP-MKLIEP
        for element in distances:
            distances_KLIEP_MKLIEP.append([element[13], element[16]])

    all_rank = []
    wins, loses, p = wilcoxon.wilcoxon(distances_LR_PLR)
    #saveCSV([[wins, loses, p]], "wilcoxon-LR-PLR", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_LR_MLR)
    #saveCSV([[wins, loses, p]], "wilcoxon-LR-MLR", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KMM_PKMM)
    #saveCSV([[wins, loses, p]], "wilcoxon-KMM-PKMM", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KMM_MKMM)
    #saveCSV([[wins, loses, p]], "wilcoxon-KMM-MKMM", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KDE_PKDE)
    #saveCSV([[wins, loses, p]], "wilcoxon-KDE-PKDE", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KDE_MKDE)
    #saveCSV([[wins, loses, p]], "wilcoxon-KDE-MKDE", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KLIEP_PKLIEP)
    #saveCSV([[wins, loses, p]], "wilcoxon-KLIEP-PKLIEP", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    wins, loses, p = wilcoxon.wilcoxon(distances_KLIEP_MKLIEP)
    #saveCSV([[wins, loses, p]], "wilcoxon-KLIEP-MKLIEP", "regression", "regression",  percentage, headers=["wins", "loses", "p-value"])
    all_rank.append([wins, loses, p])
    saveCSV(all_rank, "wilcoxon-all", "regression", "regression", percentage, headers=["wins", "loses", "p-value"])
    return all_rank
