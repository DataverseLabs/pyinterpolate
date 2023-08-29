import cProfile

import numpy as np

from pyinterpolate import read_txt
from pyinterpolate.variogram.empirical.semivariance import directional_semivariogram


DEM = read_txt('../samples/pl_dem_epsg2180.txt', skip_header=False)
DEM = DEM[:200, :]
MAX_RANGE = 15_000
STEP_SIZE = 1_500
LAGS = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)
ANGLE = 75
TOLERANCE = 0.4


def profile_directional_variogram_ellipse():
    _ = directional_semivariogram(DEM, LAGS, STEP_SIZE, None, ANGLE, TOLERANCE, method='e')
    return 0


def profile_directional_variogram_triangle():
    _ = directional_semivariogram(DEM, LAGS, STEP_SIZE, None, ANGLE, TOLERANCE, method='t')
    return 0


if __name__ == '__main__':
    cProfile.run('profile_directional_variogram_ellipse()', filename='profile_directional_variogram_ellipse.profile')
    cProfile.run('profile_directional_variogram_triangle()', filename='profile_directional_variogram_triangle.profile')