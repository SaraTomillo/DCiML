import argparse

import numpy as np

from utils.methods.friedman_nemeny import FriedmanNemenyi
from utils import IO

"""
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
    # for plancton
    #filename = "./results-total-final/" + dataset_name + "/" + dataset_name +"-"+str(seed) + "-1.csv"
    #data = pd.read_csv(filename, header = 0)
    #data = data.values

    distances = calculateDistances(data)
    IO.saveCSV(distances, "distances", dataset_name, percentage, seed)

    if np.isnan(distances).any():
        ranking = []
    else:
        ranking = friedman(distances, dataset_name, percentage)
    IO.saveCSV(ranking, "friedman", dataset_name, percentage, seed)

"""

def friedman(evals):
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
    aux = FN.getCDs()
    return data, FN.getAvgRanks(), FN.getCDs()

def friedman_pairs(evals):
    ranks = []
    for row in evals:
        aux = row[0][0] - row[0][1]
        if aux < 0:
            ranks.append([1., 2.])
        if aux > 0:
            ranks.append([2., 1.])
        if aux == 0:
            ranks.append([1.5, 1.5])
    data = []
    for rank in ranks:
        data.append(rank)
    data.append([])

    transposed_ranks = np.transpose(ranks)
    average = [np.average(transposed_ranks[0]), np.average(transposed_ranks[1])]
    data.append(average)
    return data
