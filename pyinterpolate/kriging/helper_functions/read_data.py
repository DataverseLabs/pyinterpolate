import numpy as np


def read_data(datafile, sep=','):
    with open(datafile) as f:
        x1 = f.readlines()
        x2 = [x.strip().split(sep) for x in x1]
        x3 = [[float(x[0]), float(x[1]), float(x[2])] for x in x2]
    return np.array(x3)
