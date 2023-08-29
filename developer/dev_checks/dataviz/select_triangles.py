import numpy as np
from matplotlib import pyplot as plt

from pyinterpolate.processing.select_values import select_points_within_triangle, generate_triangles


if __name__ == '__main__':

    points = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 2],
            [2, 0],
            [2, 1]
        ])

    # TRIANGLE
    triangle = (
        (-2, -2), (2, -2), (0, 4)
    )

    result = select_points_within_triangle(triangle, points)

    pr = points[result]
    tarr = np.array(triangle)
    plt.figure()
    plt.scatter(tarr[:, 0], tarr[:, 1])
    plt.scatter(pr[:, 0], pr[:, 1], c='red')
    plt.show()

    # generate triangles
    triangles = generate_triangles(points, 10, 350, tolerance=0.8)

    tarr = np.array(triangles[0])
    invtarr = np.array(triangles[1])
    plt.figure()

    plt.scatter(tarr[:, 0], tarr[:, 1])
    plt.scatter(points[0][0], points[0][1])
    plt.scatter(invtarr[:, 0], invtarr[:, 1])
    plt.show()
