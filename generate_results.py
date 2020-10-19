from distances_calculator import joinResults, calculateDistances
import rankings_regression, rankings_classification, rankings_plankton
import utils.IO as IO
import argparse

def main(datasets, problem):
    percentage = 0.33
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", help="the name of the dataset")
    parser.add_argument("percentage", help="the test percentage")
    parser.add_argument("type", help="the type of problem (regression, classification or plankton)")
    args = parser.parse_args()

    datasets = str(args.datases)
    percentage = float(args.percentage)
    problem = str(args.type)
    """
    for dataset in datasets:
        data = joinResults(dataset, percentage)
        distances = calculateDistances(data)
        IO.saveCSV(distances, "distances", dataset, percentage)

    if problem == "regression":
        generate_results_regression(datasets, percentage)
    elif problem == "classification":
        generate_results_classification(datasets, percentage)
    elif problem == "plankton":
        generate_results_plankton(datasets, percentage)


def generate_results_regression(datasets, percentage):
    ranks_methods, CDs_methods = rankings_regression.friedman_methods(datasets, percentage)
    ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M = rankings_regression.friedman_all(datasets, percentage)
    ranks_triplets_P, ranks_triplets_M, CDs_triplets = rankings_regression.friedman_triplets(datasets, percentage)
    wilcoxon = rankings_regression.wilcoxon_rank(datasets, percentage)
    IO.printResultsRegression(datasets, ranks_methods, CDs_methods, ranks_all, CDs_all, ranks_all_no_M, CDs_all_no_M , ranks_triplets_P, ranks_triplets_M, CDs_triplets, wilcoxon)

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
    datasets_regression = ["abalone", "computer-hardware", "wine-quality-red", "wine-quality-white",
                           "auto-mpg", "autos", "residential-v9", "residential-v10", "ticdata", "student-mat",
                           "student-por"]
    datasets_classification = ["iris", "sonar", "ionosphere", "cmc", "haberman",
                               "transfusion", "wdbc", "SPECT", "titanic", "splice"]
    datasets_plankton = ["plancton-2006", "plancton-2007", "plancton-2008", "plancton-2009", "plancton-2010", "plancton-2011",
                "plancton-2012", "plancton-2013", "plancton-2006-2007", "plancton-2007-2008", "plancton-2008-2009",
                "plancton-2009-2010", "plancton-2010-2011", "plancton-2011-2012",
                "plancton-2012-2013"]

    main(datasets_regression, "regression")
    main(datasets_classification, "classification")
    main(datasets_plankton, "plankton")

