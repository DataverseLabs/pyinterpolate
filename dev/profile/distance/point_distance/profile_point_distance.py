from numpy import random as nrandom
from numpy import arange
from scalene import scalene_profiler
from pyinterpolate.distance.point import point_distance


def profile_point_distance():
    sizes = arange(1, 25000, 1000)
    ds = {}
    for size in sizes:
        points = nrandom.rand(size, 2)
        other = nrandom.rand(size, 2)
        pds = point_distance(points=points, other=other)
        ds[size] = pds.nbytes / 1000000

    return ds


if __name__ == '__main__':
    scalene_profiler.start()
    results = profile_point_distance()
    scalene_profiler.stop()

    import csv

    with open('profile_point_distance_size_vs_memory.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['size', 'memory'])
        for _size, _memory in results.items():
            w.writerow([_size, _memory])
