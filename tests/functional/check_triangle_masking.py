import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pyinterpolate.distance.angular import filter_triangles_mask, \
    generate_triangles, triangle_mask


def plot_points_within_triangle(other_points,
                                coord_index,
                                lag,
                                step_size,
                                theta,
                                tolerance):

    trs_a = generate_triangles(other_points,
                               lag,
                               theta,
                               tolerance)
    trs_b = generate_triangles(other_points,
                               lag + step_size,
                               theta,
                               tolerance)
    trs_c = generate_triangles(other_points,
                               lag + step_size*2,
                               theta,
                               tolerance)

    center = other_points[coord_index]

    mask_a = triangle_mask(
        triangle_1=trs_a[coord_index][0],
        triangle_2=trs_a[coord_index][1],
        coordinates=other_points
    )
    mask_b = triangle_mask(
        triangle_1=trs_b[coord_index][0],
        triangle_2=trs_b[coord_index][1],
        coordinates=other_points
    )
    mask_c = triangle_mask(
        triangle_1=trs_c[coord_index][0],
        triangle_2=trs_c[coord_index][1],
        coordinates=other_points
    )

    mask = filter_triangles_mask(old_mask=mask_a, new_mask=mask_b)
    mask_2 = filter_triangles_mask(old_mask=mask_b, new_mask=mask_c)

    selected = other_points[mask]
    selected_2 = other_points[mask_2]

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(x=other_points[:, 0], y=other_points[:, 1])
    ax[0].scatter(x=center[0], y=center[1])
    ax[0].scatter(x=selected[:, 0], y=selected[:, 1], c='red')

    ax[1].scatter(x=other_points[:, 0], y=other_points[:, 1])
    ax[1].scatter(x=center[0], y=center[1])
    ax[1].scatter(x=selected_2[:, 0], y=selected_2[:, 1], c='red')

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
    _step_size = 0.2
    _theta = 90
    _tolerance = 0.25

    plot_points_within_triangle(
        other_points=coords,
        coord_index=int((len(coordinates)**2) / 2) + 4,
        lag=_lag,
        step_size=_step_size,
        theta=_theta,
        tolerance=_tolerance
    )
