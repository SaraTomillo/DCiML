import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

datasets_regression = []
datasets_classification = ["plankton-2006", "plankton-2007", "plankton-2008", "plankton-2009", "plankton-2010", "plankton-2011", "plankton-2012", "plankton-2013"]

datasets = []
for element in datasets_classification:
    datasets.append(element)

for element in datasets_regression:
    datasets.append(element)

f = open("makefile-plankton", "w")
f.write("python=python3\n\n")

# Write the necessary files
f.write("files =")

# Files datasets
test_percentage = 1.0
seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
for seed in seeds:
    for dataset in datasets:
        f.write(" results/error_estimations" + dataset + "/" + dataset +"-" + str(seed)+"-"+ str(test_percentage) + ".csv ")

    for i in range(2006, 2013):
        dataset = "plankton-" + str(i) + "-" +str(i+1)
        f.write(" results/error_estimations" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv ")

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

#FOR plankton
for seed in seeds:
    for dataset in datasets:
        # Write individual instructions
        f.write("results/error_estimations" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: classification_plankton.py datasets/" + dataset + "-1.csv datasets/" + dataset + "-2.csv\n")
        f.write("\t$(python) classification_plankton.py datasets/" + dataset + "-1.csv datasets/" + dataset + "-2.csv " + dataset + " 1 " + str(seed) + "\n\n")

    for i in range(2006, 2013):
        dataset = "plankton-" + str(i) + "-" +str(i+1)
        dataset_train = "datasets/plankton-" + str(i) + "-2.csv"
        dataset_test = " datasets/plankton-" + str(i+1) + "-1.csv"

        # Write individual instructions
        f.write("results/error_estimations" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: classification_plankton.py "+ dataset_train + " " + dataset_test + "\n")
        f.write("\t$(python) classification_plankton.py " + dataset_train + " " + dataset_test +" "+dataset + " 1 " + str(seed) + "\n\n")


f.write("\n\t$(python) generateResults.py\n\n")

f.close()
