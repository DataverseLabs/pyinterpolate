import numpy as np
from pyinterpolate.transform.select_points import select_kriging_data


def test_1():
    unknown_positions = np.array([
        [1, 1],
        [10, 10]
    ])
    known_positions_coordinates_xy = np.arange(0, 20)
    vals = np.ones(20)
    known_positions = np.array(
        list(
            zip(
                known_positions_coordinates_xy, known_positions_coordinates_xy, vals
            )
        )
    )

    for uk in unknown_positions:
        _ = select_kriging_data(
            unknown_position=uk,
            data_array=known_positions,
            neighbors_range=5,
            number_of_neighbors=4
        )

    assert True
