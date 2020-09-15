import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

datasets_regression = []
datasets_classification = ["plancton-2006", "plancton-2007", "plancton-2008", "plancton-2009", "plancton-2010", "plancton-2011", "plancton-2012", "plancton-2013"]

datasets = []
for element in datasets_classification:
    datasets.append(element)

for element in datasets_regression:
    datasets.append(element)

f = open("makefile-plancton", "w")
f.write("python=python3\n\n")

# Write the necessary files
f.write("files =")

# Files datasets
test_percentages = [0.33]
tests_number = 5
seeds = [2032, 2033, 2034, 2035, 2036]


# FOR PLANCTON
test_percentage = 1.0
seeds = [2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041]
for seed in seeds:
    for dataset in datasets:
        #f.write(" results/" + dataset + "/" + dataset +"-" + str(seed)+"-"+ str(test_percentage) + ".csv ")
        f.write(" results/" + dataset + "/" + dataset +"-" + str(seed) +"-" + str(test_percentage) + "-friedman.csv")

    for i in range(2006, 2013):
        dataset = "plancton-" + str(i) + "-" +str(i+1)
        #f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv ")
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv ")

f.write(" results-final/plancton/plancton-friedman-CV-KDE-MKDE-" + str(test_percentage) + ".csv ")
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

#FOR PLANCTON
for seed in seeds:
    for dataset in datasets:
        f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv")
        f.write("\n\t$(python) friedman.py " + dataset +" " + str(test_percentage) + " " + str(seed) +"\n\n")
        # Write individual instructions
        #f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: classification.py datasets-plancton/" + dataset + "-1.csv datasets-plancton/" + dataset + "-2.csv\n")
        #f.write("\t$(python) classification.py datasets-plancton/" + dataset + "-1.csv datasets-plancton/" + dataset + "-2.csv " + dataset + " 1 " + str(seed) + "\n\n")

    for i in range(2006, 2013):
        dataset = "plancton-" + str(i) + "-" +str(i+1)
        dataset_train =  "datasets-plancton/plancton-" + str(i) + "-2.csv"
        dataset_test = " datasets-plancton/plancton-" + str(i+1) + "-1.csv"

        f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv")
        f.write("\n\t$(python) friedman.py "  + dataset +" " + str(test_percentage) + " " + str(seed) +"\n\n")

        # Write individual instructions
        #f.write("results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + ".csv\t: classification.py "+ dataset_train + " " + dataset_test + "\n")
        #f.write("\t$(python) classification.py " + dataset_train + " " + dataset_test +" "+dataset + " 1 " + str(seed) + "\n\n")


f.write("results-final/plancton/plancton-friedman-CV-KDE-MKDE-" + str(test_percentage) + ".csv\t: planctonRanking.py ")
for seed in seeds:
    for dataset in datasets:
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv")
    for i in range(2006, 2013):
        dataset = "plancton-" + str(i) + "-" +str(i+1)
        f.write(" results/" + dataset + "/" + dataset + "-" + str(seed) + "-" + str(test_percentage) + "-friedman.csv ")

f.write("\n\t$(python) planctonRanking.py\n\n")

f.close()
