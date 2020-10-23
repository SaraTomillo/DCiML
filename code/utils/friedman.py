import argparse

import numpy as np

from code.utils.methods.friedman_nemeny import FriedmanNemenyi
from code.utils import IO


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
"""
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

"""

datasets = ["plancton-2006", "plancton-2007", "plancton-2008", "plancton-2009", "plancton-2010", "plancton-2011", "plancton-2012", "plancton-2013",
                           "plancton-2006-2007", "plancton-2007-2008", "plancton-2008-2009", "plancton-2009-2010", "plancton-2010-2011", "plancton-2011-2012", "plancton-2012-2013"]
percentage = 1
seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
seed = 2041
for dataset_name in datasets:
    filename = "./results-final-total/" + dataset_name + "/" + dataset_name +"-results-total-placton-1.csv"
    data = pd.read_csv(filename, sep=",", header=0, index_col=False)
    aux = data.values
    print(filename)
    print(data.head())
    data = []
    for i in range(len(aux)):
        row = []
        for element in aux[i]:
            row.append(float(element))
        data.append(row)
    distances = calculateDistances(data)

    IO.saveCSV(distances, "distances", dataset_name, percentage, seed)
"""