import argparse
import os
import utils.IO as IO
import utils.wilcoxon as wilcoxon
import utils.friedman as friedman
import pandas as pd

def main(problem):
    percentage = 0.33
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="the type of problem (regression, classification or plankton)")
    parser.add_argument("percentage", help="the test percentage")
    args = parser.parse_args()

    problem = str(args.problem)
    percentage = float(args.percentage)
    """
    plankton = False
    if problem == "regression":
        filename = "results/rankings/regression/regression/regression-friedman-triplets-M-0.33.csv"
    elif problem == "classification":
        filename = "results/rankings/classification/classification/classification-friedman-methods-0.33.csv"
    elif problem == "plankton":
        filename = "results/rankings/plankton/plankton/plankton-friendman-methods-1.csv"
        plankton = True

    LR_data, KMM_data, KDE_data, KLIEP_data = retrieveData(filename, plankton)
    CDs = generateCD(LR_data)
    print(CDs)

    wins, loses, p = generateWilcoxon(LR_data)
    print(str(wins) + "," + str(loses) + "," + str(p))
    wins, loses, p = generateWilcoxon(KMM_data)
    print(str(wins) + "," + str(loses) + "," + str(p))
    wins, loses, p = generateWilcoxon(KDE_data)
    print(str(wins) + "," + str(loses) + "," + str(p))
    wins, loses, p = generateWilcoxon(KLIEP_data)
    print(str(wins) + "," + str(loses) + "," + str(p))


def retrieveData(filename, plankton):
    filename = filename
    data = pd.read_csv(filename, sep=",", header=0)
    if plankton:
        LR_data = data[['CCV', 'LR', 'PLR']]
        KMM_data = []
        KDE_data = data[['CCV.1', 'KDE', 'PKDE']]
        KLIEP_data = []
    else:
        LR_data = data[['CCV', 'LR', 'PLR']]
        KMM_data = data[['CCV.1', 'KMM', 'PKMM']]
        KDE_data = data[['CCV.2', 'KDE', 'PKDE']]
        KLIEP_data = data[['CCV.3', 'KLIEP', 'PKLIEP']]

    return LR_data, KMM_data, KDE_data, KLIEP_data

def generateCD(triplet):
    ranks, avgRank, CDs = friedman.friedman(triplet.values)
    return CDs

def generateWilcoxon(triplet):
    wins, loses, p = wilcoxon.wilcoxon(triplet.values)
    return wins, loses, p

main("regression")
#main("classification")
#main("plankton")