import numpy as np

from pyinterpolate.idw.idw import inverse_distance_weighting


def test_idw_2d():
    unknown_pos = (10, 10)
    pos1 = [[11, 1, 1], [23, 2, 2], [33, 3, 3], [14, 44, 4], [13, 10, 9], [12, 55, 35], [11, 9, 7]]
    pos2 = [[11, 1, 1], [23, 2, 2], [33, 3, 3], [14, 44, 4], [10, 10, 999], [12, 55, 35], [11, 9, 7]]

    u_val1 = inverse_distance_weighting(np.array(pos1),
                                        unknown_pos,
                                        -1, 0.5)

    u_val2 = inverse_distance_weighting(np.array(pos1),
                                        np.array(unknown_pos),
                                        3)

    u_val3 = inverse_distance_weighting(np.array(pos2),
                                        np.array(unknown_pos),
                                        3)

    # Test case 1: u_val1 > u_val2 > 7

    assert u_val1 > u_val2
    assert u_val2 > 7

    # Test case 2: 999

    assert u_val3 == 999


def test_multidimensional_idw():
    coords = [[0, 0, 0, 1],
              [1, 1, 1, 1],
              [2, 2, 2, 1],
              [5, 5, 5, 1]]

    coords = np.array(coords)
    u_coords = [[0, 1, 2]]

    res = inverse_distance_weighting(coords, u_coords, 2, 10)

    assert res == 1
