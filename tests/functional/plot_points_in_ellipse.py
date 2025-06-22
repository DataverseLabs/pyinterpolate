import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pyinterpolate.distance.angular import select_points_within_ellipse, \
    define_whitening_matrix


def plot_points_within_ellipse(ellipse_center,
                               other_points,
                               lag,
                               step_size,
                               theta,
                               tolerance):
    w_matrix = define_whitening_matrix(theta=theta,
                                       minor_axis_size=tolerance)
    pts = select_points_within_ellipse(
        ellipse_center=ellipse_center,
        other_points=other_points,
        lag=lag,
        step_size=step_size,
        w_matrix=w_matrix
    )

    selected = other_points[pts]

    plt.figure()
    plt.scatter(x=other_points[:, 0], y=other_points[:, 1])
    plt.scatter(x=ellipse_center[0], y=ellipse_center[1])
    plt.scatter(x=selected[:, 0], y=selected[:, 1])
    plt.show()


if __name__ == '__main__':
    coordinates = np.linspace(0, 1, 20)
    coords = []
    for vx in coordinates:
        for vy in coordinates:
            coords.append([vx, vy])

    coords = np.array(coords)
    _center = np.array((0.5, 0.5))
    _lag = 0.4
    _step_size = 0.1
    _theta = 90
    _tolerance = 0.25

    plot_points_within_ellipse(
        ellipse_center=_center,
        other_points=coords,
        lag=_lag,
        step_size=_step_size,
        theta=_theta,
        tolerance=_tolerance
    )
