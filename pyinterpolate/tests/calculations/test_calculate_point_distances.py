import numpy as np
from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance


def test_calc_point_to_point_distance():
    coords = [(35.0456, -85.2672),
              (35.1174, -89.9711),
              (35.9728, -83.9422),
              (36.1667, -86.7833)]

    d = calc_point_to_point_distance(coords)

    test_arr = np.array([[0., 4.70444794, 1.6171966, 1.88558331],
                         [4.70444794, 0., 6.0892811, 3.35605413],
                         [1.6171966, 6.0892811, 0., 2.84770898],
                         [1.88558331, 3.35605413, 2.84770898, 0.]])

    assert (d.astype(int) == test_arr.astype(int)).all()


if __name__ == '__main__':
    test_calc_point_to_point_distance()
