import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

datasets_classification = ["iris", "sonar", "ionosphere", "cmc", "haberman",
                           "transfusion", "wdbc", "SPECT", "titanic", "splice"]

datasets_classification = ["haberman"]#,"cmc"] "iris", "sonar", "ionosphere",

datasets_regression = ["abalone", "computer-hardware", "wine-quality-red", "wine-quality-white",
                       "auto-mpg", "autos", "residential-v9", "residential-v10", "ticdata","student-mat","student-por"]
datasets_regression = ["abalone", "computer-hardware", "autos", "autos-mpg", "wine-quality-red"]

datasets_plankton = ["plankton-2006", "plankton-2007", "plankton-2008", "plankton-2009", "plankton-2010", "plankton-2011", "plankton-2012", "plankton-2013"]

classification_script = "classification_logReg_grid_search.py"
regression_script = "regression_logReg_grid_search.py"
plankton_script = "classification_plankton_ensemble_train_KMM_grid_search.py"

problem_type = "-LR-GS"
results_folder = problem_type + "/"

datasets = []
for element in datasets_classification:
    datasets.append(element)

for element in datasets_regression:
    datasets.append(element)

f = open("makefile", "w")#open("makefile"+problem_type, "w")
f.write("python=python3\n\n")

# Write the necessary files for regression
f.write("regressionFiles =")

# Files datasets
test_percentages = [0.33]
tests_number = 20
seeds = [2032, 2033, 2034, 2035, 2036]

for seed in seeds:
    for dataset in datasets_regression:
        for test_percentage in test_percentages:
            for test_number in range(tests_number):
                execution = str(seed)+"-"+str(test_percentage) + "-" + str(test_number) + ".csv"
                f.write(" results/error_estimations/regression" + results_folder + dataset + "/" + dataset + "-" + execution)

f.write(" results/results-final/regression"+problem_type+".csv")
f.write("\n\n")


# Write the necessary files for classification
f.write("classificationFiles =")

# Files datasets
test_percentages = [0.33]
tests_number = 20
seeds = [2033]

for seed in seeds:
    for dataset in datasets_classification:
        for test_percentage in test_percentages:
            for test_number in range(tests_number):
                execution = str(seed)+"-"+str(test_percentage) + "-" + str(test_number) + ".csv"
                f.write(" results/error_estimations/classification" + results_folder + dataset + "/" + dataset + "-" + execution)

f.write(" results/results-final/classification"+problem_type+".csv")
f.write("\n\n")

# Write the necessary files for plankton
f.write("planktonFiles =")

# Files datasets
test_percentage = 1
seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
for seed in seeds:
    for dataset in datasets_plankton:
        f.write(" results/error_estimations/plankton" + results_folder + dataset + "/" + dataset +"-" + str(seed)+"-"+ str(test_percentage) + ".csv ")

    for i in range(2006, 2013):
        dataset = "plankton-" + str(i) + "-" +str(i+1)
        f.write(" results/error_estimations/plankton"+ results_folder + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv ")

f.write(" results/results-final/plankton"+problem_type+".csv")
f.write("\n\n")

# Write ALL instruction
f.write("all\t: classification regression plankton\n")
f.write("\t@echo ALL done\n\n")

# Write REGRESSION instruction
f.write("regression\t: $(regressionFiles)\n")
f.write("\t@echo REGRESSION done\n\n")

# Write CLASSIFICATION instruction
f.write("classification\t: $(classificationFiles)\n")
f.write("\t@echo CLASSIFICATION done\n\n")

# Write PLANKTON instruction
f.write("plankton\t: $(planktonFiles)\n")
f.write("\t@echo PLANKTON done\n\n")

# Write DELETE instruction
f.write("delete\t:\n")
f.write("\trm -f $(files)\n")
f.write("\t@echo DELETE done\n\n")

# Write BUILD instruction
f.write("build\t: delete all\n")
f.write("\t@echo BUILD done\n\n")


seeds = [2032, 2033, 2034, 2035, 2036]
# DATASETS REGRESSION
for seed in seeds:
    for dataset in datasets_regression:
        for test_percentage in test_percentages:
            dataset_name = "results/error_estimations/regression"+ results_folder + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage)

        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write(dataset_name + "-" + str(test_number) + ".csv\t: " + regression_script + " datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + "\n")
            f.write("\t$(python) " + regression_script + " datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/regression/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + " " + dataset + " " + execution[:-4] + " " + str(seed) + "\n\n")

f.write("results/results-final/regression"+problem_type+".csv\t: generate_results.py \n")
f.write("\t$(python) generate_results.py " + problem_type + "\n\n")


# DATASETS CLASSIFICATION
for seed in seeds:
    for dataset in datasets_classification:
        for test_percentage in test_percentages:
            dataset_name = "results/error_estimations/classification" + results_folder + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage)

        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write(dataset_name + "-" + str(test_number) + ".csv\t: " + classification_script + " datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + "\n")
            f.write("\t$(python) "+ classification_script+" datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets/classification/" + dataset + "/datasets-" + str(seed) + "/" + dataset + "-test-" + execution + " " + dataset + " " + execution[:-4] + " " + str(seed) + "\n\n")

f.write("results/results-final/classification"+problem_type+".csv\t: generate_results.py \n")
f.write("\t$(python) generate_results.py classification" + problem_type + "\n\n")


#FOR plankton
test_percentage = 1
seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
for seed in seeds:
    for dataset in datasets_plankton:
        # Write individual instructions
        f.write("results/error_estimations/plankton" + results_folder + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: "+plankton_script+" datasets/plankton/" + dataset + "-1.csv datasets/plankton/" + dataset + "-2.csv\n")
        f.write("\t$(python) "+plankton_script+" datasets/plankton/" + dataset + "-1.csv datasets/plankton/" + dataset + "-2.csv " + dataset + " 1 " + str(seed) + "\n\n")

    for i in range(2006, 2013):
        dataset = "plankton-" + str(i) + "-" +str(i+1)
        dataset_train = "datasets/plankton/plankton-" + str(i) + "-2.csv"
        dataset_test = " datasets/plankton/plankton-" + str(i+1) + "-1.csv"

        # Write individual instructions
        f.write("results/error_estimations/plankton"+results_folder + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: "+plankton_script+" "+ dataset_train + " " + dataset_test + "\n")
        f.write("\t$(python) "+plankton_script +" "+ dataset_train + " " + dataset_test +" "+dataset + " 1 " + str(seed) + "\n\n")


f.write("results/results-final/plankton"+problem_type+".csv\t: generate_results.py \n")
f.write("\t$(python) generate_results.py plankton"+ problem_type + " " + str(test_percentage) +"\n\n")

f.close()
