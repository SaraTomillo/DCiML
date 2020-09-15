import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

datasets_classification = ["iris", "sonar", "ionosphere", "cmc", "haberman",
                           "transfusion", "wdbc","SPECT", "titanic","splice"]
#datasets_classification = ["sonar","SPECT", "transfusion", "haberman", "ionosphere", "wdbc"]
datasets_classification = ["iris", "titanic", "cmc", "splice"]
datasets_regression = ["abalone", "computer-hardware", "wine-quality-red", "wine-quality-white",
                       "auto-mpg", "autos", "residential-v9", "residential-v10"]
                       #"ticdata"]
                       #"student-mat","student-por"]

# FOR PLANCTON
#datasets_classification = ["plancton-2006", "plancton-2007", "plancton-2008", "plancton-2009", "plancton-2010", "plancton-2011", "plancton-2012", "plancton-2013"]

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
                #f.write(" results/"+ dataset + "/" + dataset + "-" + execution)
                f.write(" results/"+ dataset + "/" + dataset + "-" + str(seed)+"-"+str(test_percentage) + "-friedman.csv")

"""
# FOR PLANCTON
test_percentage = 1
seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
for seed in seeds:
    for dataset in datasets:
        f.write(" results/" + dataset + "/" + dataset +"-" + str(seed)+"-"+ str(test_percentage) + ".csv ")

    for i in range(2006, 2013):
        dataset = "plancton-" + str(i) + "-" +str(i+1)
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv ")
"""

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
            dataset_name = "results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage)
            f.write(dataset_name+"-friedman.csv\t: friedman.py")
            for test_number in range(tests_number):
                f.write(" " + dataset_name + "-" + str(test_number) + ".csv")
            f.write("\n\t$(python) friedman.py " + dataset + " " + str(test_percentage) + " " + str(seed) + "\n\n")
        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write(dataset_name + "-" + str(test_number) + ".csv\t: regression.py datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-test-" + execution + "\n")
            f.write("\t$(python) regression.py datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-train-" + str(
                test_percentage) + ".csv datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-test-" + execution + " " + dataset + " " + execution[:-4] + " " + str(seed) + "\n\n")

# DATASETS CLASSIFICATION
for seed in seeds:
    for dataset in datasets_classification:
        for test_percentage in test_percentages:
            dataset_name = "results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage)
            f.write(dataset_name + "-friedman.csv\t: friedman.py")
            for test_number in range(tests_number):
                f.write(" " + dataset_name + "-" + str(test_number) + ".csv")
            f.write("\n\t$(python) friedman.py " + dataset + " " + str(test_percentage) + " " + str(seed) + "\n\n")
        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write(dataset_name + "-" + str(test_number) + ".csv\t: classification.py datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-train-" + str(test_percentage) + ".csv datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-test-" + execution + "\n")
            f.write("\t$(python) classification.py datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-train-" + str(
                test_percentage) + ".csv datasets-" + str(seed) + "/" + dataset + "/" + dataset + "-test-" + execution + " " + dataset + " " + execution[:-4] + " " + str(seed) + "\n\n")

"""
# FOR PLANCTON
for seed in seeds:
    for dataset in datasets:
        f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv")
        f.write("\n\t$(python) friedman.py " + dataset + "\n\n")
        # Write individual instructions
        f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: classification.py datasets-plancton/" + dataset + "-1.csv datasets-plancton/" + dataset + "-2.csv\n")
        f.write("\t$(python) classification.py datasets-plancton/" + dataset + "-1.csv datasets-plancton/" + dataset + "-2.csv " + dataset + " 1 " + str(seed) + "\n\n")

    for i in range(2006, 2013):
        dataset = "plancton-" + str(i) + "-" +str(i+1)
        dataset_second_half = "plancton-" + str(i) + "-1"
        dataset_first_half = "plancton-" + str(i+1) + "-1"

        f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv")
        f.write("\n\t$(python) friedman.py " + dataset + "\n\n")

        # Write individual instructions
        f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: classification.py datasets-plancton/" + dataset + "-1.csv datasets-plancton/" + dataset + "-2.csv\n")
        f.write("\t$(python) classification.py " + dataset_second_half +" " + dataset_first_half + " "+dataset + " 1 " + str(seed) + "\n\n")

"""
f.close()
