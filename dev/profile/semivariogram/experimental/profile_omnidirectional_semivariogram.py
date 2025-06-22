from numpy import random as nrandom

from pyinterpolate.semivariogram.experimental import calculate_semivariance


def profile_calculate_semivariance():
    points = nrandom.rand(10000, 3)
    step_size = 0.05
    max_range = 0.6

    _ = calculate_semivariance(
        ds=points,
        step_size=step_size,
        max_range=max_range
    )


if __name__ == '__main__':
    profile_calculate_semivariance()
