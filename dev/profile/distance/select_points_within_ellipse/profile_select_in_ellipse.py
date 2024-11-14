import numpy as np
from numpy import random as nrandom
from numpy import arange
from scalene import scalene_profiler

from pyinterpolate.distance.angular import select_points_within_ellipse
from pyinterpolate.distance.point import point_distance


def profile_select_in_ellipse():
    grid = np.random.rand(500, 2)
    for pt in grid:
        theta = np.random.randint(0, 360)
        step_size = np.random.random()
        lag = np.random
        _ = select_points_within_ellipse(ellipse_center=pt,
                                         other_points=grid,
                                         lag=step_size * 2,
                                         step_size=step_size,
                                         theta=theta,
                                         minor_axis_size=0.1)


if __name__ == '__main__':
    profile_select_in_ellipse()
