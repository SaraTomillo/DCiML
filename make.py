import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


datasets_classification = ["iris", "sonar", "ionosphere", "cmc", "haberman",
                           "transfusion", "wdbc","SPECT", "titanic","splice"]

datasets_regression = ["abalone", "computer-hardware", "wine-quality-red", "wine-quality-white",
                       "auto-mpg", "autos", "residential-v9", "residential-v10", "ticdata", "student-mat","student-por"]

datasets_classification = ["iris"]
datasets_regression = []

datasets = []
for element in datasets_classification:
    datasets.append(element)

for element in datasets_regression:
    datasets.append(element)

f = open("makefile", "w")
f.write("python=python3\n\n")

# Write the necessary files
f.write("files =")

# Files datasets
test_percentages = [0.33]
tests_number = 20
seeds = [2032, 2033, 2034, 2035, 2036]

for seed in seeds:
    for dataset in datasets:
        for test_percentage in test_percentages:
            for test_number in range(tests_number):
                execution = str(seed)+"-"+str(test_percentage) + "-" + str(test_number) + ".csv"
                f.write(" results/error_estimations/"+ dataset + "/" + dataset + "-" + execution)

f.write(" results/results-final/regression.csv")
f.write(" results/results-final/classification.csv")

f.write("\n\n")

# Write ALL instruction
f.write("all\t: $(files)\n")
f.write("\t@echo ALL done\n\n")

# Write DELETE instruction
f.write("delete\t:\n")
f.write("\trm -f $(files)\n")
f.write("\t@echo DELETE done\n\n")

# Write BUILD instruction
f.write("build\t: delete all\n")
f.write("\t@echo BUILD done\n\n")

# DATASETS REGRESSION
for seed in seeds:
    for dataset in datasets_regression:
        for test_percentage in test_percentages:
            dataset_name = "results/error_estimations/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage)

        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write(dataset_name + "-" + str(test_number) + ".csv\t: regression.py datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + "\n")
            f.write("\t$(python) regression.py datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + " " + dataset + " " + execution[:-4] + " " + str(seed) + "\n\n")

# DATASETS CLASSIFICATION
for seed in seeds:
    for dataset in datasets_classification:
        for test_percentage in test_percentages:
            dataset_name = "results/error_estimations/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage)

        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write(dataset_name + "-" + str(test_number) + ".csv\t: classification.py datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + "\n")
            f.write("\t$(python) classification.py datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + " " + dataset + " " + execution[:-4] + " " + str(seed) + "\n\n")

f.write("results/results-final/regression.csv\t: generate_results.py \n")
f.write("\t$(python) generate_results.py\n\n")

f.write("results/results-final/classification.csv\t: generate_results.py \n")
f.write("\t$(python) generate_results.py\n\n")

f.close()
