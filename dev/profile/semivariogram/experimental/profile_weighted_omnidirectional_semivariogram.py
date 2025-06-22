import numpy as np
from numpy import random as nrandom

from pyinterpolate.semivariogram.experimental import calculate_semivariance


def profile_calculate_semivariance():
    size = 10000
    points = nrandom.rand(size, 3)
    weights = np.random.randint(1, 100, size)
    step_size = 0.05
    max_range = 0.6

    _ = calculate_semivariance(
        ds=points,
        step_size=step_size,
        max_range=max_range,
        custom_weights=weights
    )


if __name__ == '__main__':
    profile_calculate_semivariance()
