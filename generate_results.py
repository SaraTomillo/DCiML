import utils.IO as IO
import os
import argparse

import distances_calculator, rankings_regression, rankings_classification, rankings_plankton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="the type of problem (regression, classification or plankton)")
    parser.add_argument("percentage", help="the test percentage")
    args = parser.parse_args()

    problem = str(args.problem)
    percentage = float(args.percentage)

    if "regression" in problem:
        datasets = retrieveDatasets(problem)
        print(datasets)
        #for dataset in datasets:
        #    calculateDistances(problem, dataset, percentage)
        generate_results_regression(datasets, problem, percentage)

    elif "classification" in problem:
        datasets = retrieveDatasets(problem)
        for dataset in datasets:
            calculateDistances(problem, dataset, percentage)
        generate_results_classification(datasets, problem, percentage)

    elif "plankton" in problem:
        datasets = retrieveDatasets(problem)
        #for dataset in datasets:
        #    calculateDistances(problem, dataset, percentage)
        generate_results_plankton(datasets, problem, percentage)


def calculateDistances(problem, dataset, percentage):
    data = distances_calculator.joinResults(problem, dataset, percentage)
    distances = distances_calculator.calculateDistances(data)
    IO.saveCSV(distances, "distances", problem, dataset, percentage)

def retrieveDatasets(problem):
    datasets = []
    path = os.getcwd()+"/results/error_estimations/" + problem + "/"
    for dirpath, dirnames, files in os.walk(path):
        if dirpath != path:
            break
        datasets = dirnames
    return datasets



def generate_results_regression(datasets, problem, percentage):
    ranks_methods, CDs_methods = rankings_regression.friedman_methods(datasets, problem, percentage)
    #ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M = rankings_regression.friedman_all(datasets, problem, percentage)
    #ranks_triplets_P, ranks_triplets_M, CDs_triplets = rankings_regression.friedman_triplets(datasets, problem, percentage)
    ranks_triplets_P = []
    ranks_triplets_M = []
    CDs_triplets = []
    wilcoxon = []
    #wilcoxon = rankings_regression.wilcoxon_rank(datasets, problem, percentage)
    #IO.printResultsRegression(datasets, problem, ranks_methods, CDs_methods, ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M, ranks_triplets_P, ranks_triplets_M, CDs_triplets, wilcoxon)

def generate_results_classification(datasets, problem, percentage):
    ranks_methods, CDs_methods = rankings_classification.friedman_methods(datasets, problem, percentage)
    ranks_all, CDs_all = rankings_classification.friedman_all(datasets, problem, percentage)
    wilcoxon = rankings_classification.wilcoxon_rank(datasets, problem, percentage)
    IO.printResultsClassification(datasets, problem, ranks_methods, CDs_methods, ranks_all, CDs_all, wilcoxon)

def generate_results_plankton(datasets, problem, percentage):
    ranks_methods, CDs_methods = rankings_plankton.friedman_methods(datasets, problem, percentage)
    ranks_all, CDs_all = rankings_plankton.friedman_all(datasets, problem, percentage)
    wilcoxon = rankings_plankton.wilcoxon_rank(datasets, problem, percentage)
    IO.printResultsPlankton(datasets, problem, ranks_methods, CDs_methods, ranks_all, CDs_all, wilcoxon)

if __name__ == "__main__":
    main()
