from . import rankings_classification, rankings_regression, distances_calculator, rankings_plankton
import code.utils.IO as IO
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="the type of problem (regression, classification or plankton)")
    parser.add_argument("percentage", help="the test percentage")
    args = parser.parse_args()

    problem = str(args.problem)
    percentage = float(args.percentage)

    if problem == "regression":
        datasets = retrieveDatasets(problem)
        for dataset in datasets:
            calculateDistances(problem, dataset, percentage)
        generate_results_regression(datasets, percentage)

    elif problem == "classification":
        datasets = retrieveDatasets(problem)
        for dataset in datasets:
            calculateDistances(problem, dataset, percentage)
        generate_results_classification(datasets, percentage)

    elif problem == "plankton":
        datasets = retrieveDatasets(problem)
        for dataset in datasets:
            calculateDistances(problem, dataset, percentage)
        generate_results_plankton(datasets, percentage)


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



def generate_results_regression(datasets, percentage):
    ranks_methods, CDs_methods = rankings_regression.friedman_methods(datasets, percentage)
    ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M = rankings_regression.friedman_all(datasets, percentage)
    ranks_triplets_P, ranks_triplets_M, CDs_triplets = rankings_regression.friedman_triplets(datasets, percentage)
    wilcoxon = rankings_regression.wilcoxon_rank(datasets, percentage)
    IO.printResultsRegression(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M, ranks_triplets_P, ranks_triplets_M, CDs_triplets, wilcoxon)

def generate_results_classification(datasets, percentage):
    ranks_methods, CDs_methods = rankings_classification.friedman_methods(datasets, percentage)
    ranks_all, CDs_all = rankings_classification.friedman_all(datasets, percentage)
    wilcoxon = rankings_classification.wilcoxon_rank(datasets, percentage)
    IO.printResultsClassification(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, wilcoxon)

def generate_results_plankton(datasets, percentage):
    ranks_methods, CDs_methods = rankings_plankton.friedman_methods(datasets, percentage)
    ranks_all, CDs_all = rankings_plankton.friedman_all(datasets, percentage)
    wilcoxon = rankings_plankton.wilcoxon_rank(datasets, percentage)
    IO.printResultsPlankton(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, wilcoxon)

if __name__ == "__main__":
    main()
