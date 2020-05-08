import argparse
import os

import numpy as np
import pandas as pd

from friedman_nemeny import FriedmanNemenyi
import IO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="the name of the dataset")
    parser.add_argument("percentage", help="the test percentage")
    #parser.add_argument("test_number", help="the number of the test distribution")
    parser.add_argument("seed", help="the seed")
    args = parser.parse_args()

    dataset_name = str(args.dataset_name)
    percentage = float(args.percentage)
    #test_number = int(args.test_number)
    seed = int(args.seed)

    data = join(dataset_name, percentage, seed)
    distances = calculateDistances(data)
    IO.saveCSV(distances, "distances", dataset_name, percentage, seed)
    if np.isnan(distances).any():
        ranking = []
    else:
        ranking = friedman(distances, dataset_name, percentage)
    IO.saveCSV(ranking, "friedman", dataset_name, percentage, seed)


def friedman(evals, dataset_name, percentage):
    FN = FriedmanNemenyi(evals, order=1, decimals=4)
    data = []
    for rank in FN.getRanks():
         data.append(rank)
    data.append([])
    data.append(FN.getAvgRanks())
    data.append([])
    data.append(FN.getDeviationRanks())
    data.append([])
    data.append(FN.getCDs())
    return data

def calculateDistances(data):
    distances = []
    for row in data:
        eval = float(row[0])
        i = 0
        distance = np.zeros(len(row[1:]))
        for val in row[1:]:
            if not np.math.isnan(val):
                distance[i] = abs(eval - float(val))
            else:
                distance[i] = float("nan") #eval
            i+=1
        distances.append(distance)
    return distances


def join(dataset, percentage, seed):
    data = []
    for subdir, dirs, files in os.walk(os.getcwd()+"/results-" + str(seed) + "/" + dataset + "/"):
        for file in files:
            if str(percentage) in file:
                if not "distances" in file and not "friedman" in file:
                    filename = str(os.path.join(os.path.join("results-" + str(seed)+"/", dataset), file))
                    aux = pd.read_csv(filename, sep=",", header=0, index_col=False)
                    data.append(np.asarray(aux.iloc[0]))#.to_numpy())
    return np.asarray(data)



if __name__ == "__main__":
    main()
