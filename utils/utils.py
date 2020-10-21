import tracemalloc
import linecache
import os
from random import Random

import numpy as np


def minmax(data):
    aux = np.transpose(data)
    if len(np.shape(aux)) == 1:
        aux = [aux]

    minmax = []
    for col in aux:
        min = float('inf')
        max = -min
        for elem in col:
            if elem > max:
                max = elem
            if elem < min:
                min = elem
        new_col = [min, max]
        minmax.append(new_col)
    return minmax

def reduce(data, minmax):
    aux = np.transpose(data)
    if len(np.shape(aux))==1:
        aux = [aux]
    new_data = []
    for i in range(len(aux)):
        col = aux[i]
        min = minmax[i][0]
        max = minmax[i][1]
        ran = max - min
        if ran > 0:
            new_col = (col - min) / ran
        else:
            new_col = col
        new_data.append(new_col)
    return(np.transpose(new_data))

def npShuffle(M,seed):
    r=Random()
    r.seed(seed)
    Ml=M.tolist()
    r.shuffle(Ml)
    return np.array(Ml)

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    return total / 1024

