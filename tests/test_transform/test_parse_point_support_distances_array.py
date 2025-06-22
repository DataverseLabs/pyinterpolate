import numpy as np

from pyinterpolate.distance.point import point_distance
from pyinterpolate.transform.transform import parse_point_support_distances_array


def test_parse_1():
    points = np.array([(0, 0, 10), (1, 1, 40), (0, 1, 20)])
    other = np.array([(0, 1, -10), (2, 1, -30)])

    # rows - points
    # cols - other
    distances: np.ndarray
    distances = point_distance(points=points[:, :-1], other=other[:, :-1])
    distances = distances.astype(int)

    expected_output = np.array([
        [10, -10, 1],
        [10, -30, 2],
        [40, -10, 1],
        [40, -30, 1],
        [20, -10, 0],
        [20, -30, 2]
    ])

    parsed = parse_point_support_distances_array(
        distances=distances,
        values_a=points[:, -1],
        values_b=other[:, -1]
    )

    for idx, row in enumerate(expected_output):
        assert np.array_equal(row, parsed[idx])
