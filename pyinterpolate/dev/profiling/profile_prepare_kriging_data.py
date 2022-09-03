import cProfile as cp
import numpy as np
from pstats import Stats
from pyinterpolate.transform import prepare_kriging_data


if __name__ == '__main__':
    arr_size = 10**7
    rng = 0.9
    upos = (0.1, 0.1)
    arr = np.random.rand(arr_size, 3)
    pr = cp.Profile()
    pr.enable()
    _ = prepare_kriging_data(upos, arr, rng)
    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(5)
