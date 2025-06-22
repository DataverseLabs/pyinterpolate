import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pyinterpolate.distance.angular import select_points_within_triangle, \
    generate_triangles, triangle_mask


def plot_points_within_triangle(other_points,
                                coord_index,
                                lag,
                                theta,
                                tolerance):

    trs = generate_triangles(other_points, lag, theta, tolerance)

    center = other_points[coord_index]

    mask = triangle_mask(
        triangle_1=trs[coord_index][0],
        triangle_2=trs[coord_index][1],
        coordinates=other_points
    )

    selected = other_points[mask]

    plt.figure()
    plt.scatter(x=other_points[:, 0], y=other_points[:, 1])
    plt.scatter(x=center[0], y=center[1])
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

    plot_points_within_triangle(
        other_points=coords,
        coord_index=int((len(coordinates)**2) / 2) + 4,
        lag=_lag,
        theta=_theta,
        tolerance=_tolerance
    )
