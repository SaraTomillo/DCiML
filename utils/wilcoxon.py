import numpy as np
import scipy.stats as stats

def wilcoxon(elements):
    distances = []
    wins = []
    for each in elements:
        first = each[0]
        second = each[1]

        if first < second:
            distances.append(1.0)
            wins.append(1)
        if first > second:
            distances.append(-1.0)
            wins.append(0)
        if first == second:
            distances.append(0.0)
            wins.append(0)

    w,p = stats.wilcoxon(distances)
    loses = len(wins) - np.sum(wins)
    wins = np.sum(wins)
    return wins, loses, p