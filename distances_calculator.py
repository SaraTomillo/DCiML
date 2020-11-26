import os
import pandas as pd
import numpy as np


def joinResults(problem_type, dataset, percentage):
    data = []
    for subdir, dirs, files in os.walk(os.getcwd()+"/results/error_estimations/" + problem_type + "/" + dataset + "/"):
        for file in files:
            #if str(percentage) in file:
                if not "distances" in file and not "friedman" in file:
                    filename = str(os.path.join(os.path.join("results/error_estimations/", problem_type + "/" + dataset), file))
                    aux = pd.read_csv(filename, sep=",", header=0, index_col=False)
                    data.append(np.asarray(aux.iloc[0]))
    return np.asarray(data)


def calculateDistances(data):
    distances = []
    for row in data:
        eval = float(row[0])
        i = 0
        distance = np.zeros(len(row[1:]))
        contains_nan = False
        for val in row[1:]:
            if not np.math.isnan(val):
                distance[i] = abs(eval - float(val))
            else:
                contains_nan = True
                distance[i] = float("nan")
            i+=1
        if not contains_nan:
            distances.append(distance)
    return distances
