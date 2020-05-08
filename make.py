import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


# faltan = ["auto-mpg", "transfusion"]
#datasets = []
datasets_regression= []
#datasets_classification = ["iris", "abalone", "ionosphere", "sonar",
#                           "cmc", "haberman", "SPECT", "wdbc", "transfusion"]
#datasets_classification = ["statlog", "semeion"]

datasets_regression = ["computer-hardware"]
#datasets_regression = ["wine-quality-red", "bikes-day-casual", "bikes-day-registered", "bikes-day-total"]

#                      "bikes-hour-casual", "bikes-hour-registered", "bikes-hour-total"]

#"wine-quality-white",
#"computer-hardware"

datasets = []
for element in datasets_regression:
    datasets.append(element)

f = open("makefile", "w")
f.write("python=python3\n\n")

seed = 2032

# Write the necessary files
f.write("files =")

# Files datasets
test_percentages = [0.33]
tests_number = 5
for dataset in datasets:
    for test_percentage in test_percentages:
        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            f.write(" results-" + str(seed)+"/"+dataset + "/" + dataset + "-" + execution)
            f.write(" results-" + str(seed)+"/"+dataset + "/" + dataset + "-" + str(test_percentage) + "-friedman.csv")

"""
# Files iris
test_percentages = [0.33]
tests_number = 20
for test_percentage in test_percentages:
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        f.write(" results-" + str(seed)+"/iris/iris-" + execution)
        f.write(" results-" + str(seed)+"/iris/iris-" + str(test_percentage) + "-friedman.csv")


# Files abalone
test_percentages = [0.33]
tests_number = 20
for test_percentage in test_percentages:
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        f.write(" results-" + str(seed)+"/abalone/abalone-" + execution)
        f.write(" results-" + str(seed)+"/abalone/abalone-" + str(test_percentage) + "-friedman.csv")


# Files ionosphere
test_percentages = [0.33]
tests_number = 20
for test_percentage in test_percentages:
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        f.write(" results-" + str(seed)+"/ionosphere/ionosphere-" + execution)
        f.write(" results-" + str(seed)+"/ionosphere/ionosphere-" + str(test_percentage) + "-friedman.csv")

# Files sonar
test_percentages = [0.33]
tests_number = 5
for test_percentage in test_percentages:
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        f.write(" results-" + str(seed)+"/sonar/sonar-" + execution)
        f.write(" results-" + str(seed)+"/sonar/sonar-" + str(test_percentage) + "-friedman.csv")

# Files auto-mpg
test_percentages = [0.33]
tests_number = 20
for test_percentage in test_percentages:
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        f.write(" results-" + str(seed)+"/auto-mpg/auto-mpg-" + execution)
        f.write(" results-" + str(seed)+"/auto-mpg/auto-mpg-" + str(test_percentage) + "-friedman.csv")


# Files spambase
test_percentages = [0.3]
tests_number = 5
for test_percentage in test_percentages:
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        f.write(" results-" + str(seed)+"/spambase/spambase-" + execution)
        f.write(" results-" + str(seed)+"/spambase/spambase-" + str(test_percentage) + "-friedman.csv")
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


"""
# IRIS DATASET
test_percentages = [0.33]
tests_number = 20
for test_percentage in test_percentages:
    # Write friedman instruction
    f.write("results-" + str(seed)+"/iris/iris-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
    for test_number in range(tests_number):
        f.write(" results-" + str(seed)+"/iris/iris-" + str(test_percentage) + "-"+ str(test_number) +".csv")
    f.write("\n\t$(python) friedman.py iris " + str(test_percentage) +" " + str(seed)+ "\n\n")
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        # Write individual instructions
        f.write("results-" + str(seed)+"/iris/iris-" + execution + "\t: classification.py datasets-" + str(seed)+"/iris/iris-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/iris/iris-test-"+ execution +"\n")
        f.write("\t$(python) classification.py datasets-" + str(seed)+"/iris/iris-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/iris/iris-test-"+ execution +" iris "+  execution[:-4]+" " + str(seed)+ "\n\n")


# ABALONE DATASET
test_percentages = [0.33]
tests_number = 20

for test_percentage in test_percentages:
    f.write("results-" + str(seed)+"/abalone/abalone-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
    for test_number in range(tests_number):
        f.write(" results-" + str(seed)+"/abalone/abalone-" + str(test_percentage) + "-" + str(test_number) + ".csv")
    f.write("\n\t$(python) friedman.py abalone " + str(test_percentage) +" " + str(seed)+  "\n\n")
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        # Write individual instructions
        f.write("results-" + str(seed)+"/abalone/abalone-" + execution + "\t: regression.py datasets-" + str(seed)+"/abalone/abalone-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/abalone/abalone-test-"+ execution +"\n")
        f.write("\t$(python) regression.py datasets-" + str(seed)+"/abalone/abalone-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/abalone/abalone-test-"+ execution +" abalone "+  execution[:-4]+" " + str(seed)+ "\n\n")

# IONOSPHERE DATASET
test_percentages = [0.33]
tests_number = 20

for test_percentage in test_percentages:
    f.write("results-" + str(seed)+"/ionosphere/ionosphere-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
    for test_number in range(tests_number):
        f.write(" results-" + str(seed)+"/ionosphere/ionosphere-" + str(test_percentage) + "-" + str(test_number) + ".csv")
    f.write("\n\t$(python) friedman.py ionosphere " + str(test_percentage) + " " + str(seed)+ "\n\n")
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        # Write individual instructions
        f.write("results-" + str(seed)+"/ionosphere/ionosphere-" + execution + "\t: classification.py datasets-" + str(seed)+"/ionosphere/ionosphere-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/ionosphere/ionosphere-test-"+ execution +"\n")
        f.write("\t$(python) classification.py datasets-" + str(seed)+"/ionosphere/ionosphere-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/ionosphere/ionosphere-test-"+ execution +" ionosphere "+  execution[:-4]+" " + str(seed)+ "\n\n")

# SONAR DATASET
test_percentages = [0.33]
tests_number = 20

for test_percentage in test_percentages:
    f.write("results-" + str(seed)+"/sonar/sonar-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
    for test_number in range(tests_number):
        f.write(" results-" + str(seed)+"/sonar/sonar-" + str(test_percentage) + "-" + str(test_number) + ".csv")
    f.write("\n\t$(python) friedman.py sonar " + str(test_percentage) + " " + str(seed)+ "\n\n")
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        # Write individual instructions
        f.write("results-" + str(seed)+"/sonar/sonar-" + execution + "\t: classification.py datasets-" + str(seed)+"/sonar/sonar-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/sonar/sonar-test-"+ execution +"\n")
        f.write("\t$(python) classification.py datasets-" + str(seed)+"/sonar/sonar-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/sonar/sonar-test-"+ execution +" sonar "+  execution[:-4]+" " + str(seed)+ "\n\n")


# AUTO-MPG DATASET
test_percentages = [0.33]
tests_number = 20

for test_percentage in test_percentages:    # Write friedman instruction
    f.write("results-" + str(seed)+"/auto-mpg/auto-mpg-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
    for test_number in range(tests_number):
        f.write(" results-" + str(seed)+"/auto-mpg/auto-mpg-" + str(test_percentage) + "-" + str(test_number) + ".csv")
    f.write("\n\t$(python) friedman.py auto-mpg " + str(test_percentage) + " " + str(seed)+ "\n\n")
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        # Write individual instructions
        f.write("results-" + str(seed)+"/auto-mpg/auto-mpg-" + execution + "\t: regression.py datasets-" + str(seed)+"/auto-mpg/auto-mpg-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/auto-mpg/auto-mpg-test-"+ execution +"\n")
        f.write("\t$(python) regression.py datasets-" + str(seed)+"/auto-mpg/auto-mpg-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/auto-mpg/auto-mpg-test-"+ execution +" auto-mpg "+  execution[:-4]+" " + str(seed)+ "\n\n")


# SPAMBASE DATASET
test_percentages = [0.3]
tests_number = 5

for test_percentage in test_percentages:
    # Write friedman instruction
    f.write("results-" + str(seed)+"/spambase/spambase-" + str(test_percentage) + "-friedman.csv\t: friedman.py results-" + str(seed)+"/spambase/spambase-" + str(test_percentage) + ".csv\n")
    f.write("\t$(python) friedman.py spambase " + str(test_percentage) + "\n\n")
    for test_number in range(tests_number):
        execution = str(test_percentage) + "-" + str(test_number) + ".csv"
        # Write individual instructions
        f.write("results-" + str(seed)+"/spambase/spambase-" + execution + "\t: classification.py datasets-" + str(seed)+"/spambase/spambase-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/spambase/spambase-test-"+ execution +"\n")
        f.write("\t$(python) classification.py datasets-" + str(seed)+"/spambase/spambase-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/spambase/spambase-test-"+ execution +" spambase "+  execution[:-4]+"\n\n")
"""
# DATASETS REGRESSION
test_percentages = [0.33]
tests_number = 20
for dataset in datasets_regression:
    for test_percentage in test_percentages:
        f.write("results-" + str(seed)+"/" + dataset + "/" + dataset +"-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
        for test_number in range(tests_number):
            f.write(" results-" + str(seed)+"/" + dataset + "/" + dataset +"-"+ str(test_percentage) + "-" + str(test_number) + ".csv")
        f.write("\n\t$(python) friedman.py "+dataset+" "+ str(test_percentage) + " " + str(seed)+ "\n\n")
        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write("results-" + str(seed)+"/" + dataset + "/" + dataset +"-" + execution + "\t: classification.py datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-test-"+ execution +"\n")
            f.write("\t$(python) regression.py datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-test-"+ execution +" "+dataset+" "+  execution[:-4]+" " + str(seed)+ "\n\n")
"""


# DATASETS CLASSIFICATION
test_percentages = [0.33]
tests_number = 20
for dataset in datasets_classification:
    for test_percentage in test_percentages:
        f.write("results-" + str(seed)+"/" + dataset + "/" + dataset +"-" + str(test_percentage) + "-friedman.csv\t: friedman.py")
        for test_number in range(tests_number):
            f.write(" results-" + str(seed)+"/" + dataset + "/" + dataset +"-"+ str(test_percentage) + "-" + str(test_number) + ".csv")
        f.write("\n\t$(python) friedman.py " + dataset + " " + str(test_percentage) + " " + str(seed)+ "\n\n")
        for test_number in range(tests_number):
            execution = str(test_percentage) + "-" + str(test_number) + ".csv"
            # Write individual instructions
            f.write("results-" + str(seed)+"/" + dataset + "/" + dataset +"-" + execution + "\t: classification.py datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-test-"+ execution +"\n")
            f.write("\t$(python) classification.py datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-train-" + str(test_percentage) + ".csv datasets-" + str(seed)+"/" + dataset + "/" + dataset +"-test-"+ execution +" "+dataset+" "+  execution[:-4]+" " + str(seed)+ "\n\n")

"""
f.close()
