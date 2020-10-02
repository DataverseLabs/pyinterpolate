import numpy as np
from pyinterpolate.calculations.distances.calculate_distances import calc_block_to_block_distance


def test_calc_block_to_block_distance():
    # Test block_to_block_distances

    areal_test_data = [
        ['id0', [[0, 1, 1],
                 [0, 2, 1],
                 [0, 3, 1]]
         ],
        ['id1', [[1, 0, 2],
                 [1, 1, 2]]],

        ['id2', [[2, 1, 3],
                 [2, 2, 4],
                 [2, 3, 4]]]
    ]

    x = calc_block_to_block_distance(areal_test_data)
    assert (int(np.sum(x[0])) == 12)


if __name__ == '__main__':
    test_calc_block_to_block_distance()
