import numpy as np

from pyinterpolate.transform.select_points import select_kriging_data_from_direction


def profile_select_directional():
    unknown_positions = np.random.random(size=(1000, 2))
    known_positions = np.random.random(size=(5000, 3))

    for uk in unknown_positions:
        _ = select_kriging_data_from_direction(
            unknown_position=uk,
            data_array=known_positions,
            neighbors_range=0.1,
            direction=15,
            number_of_neighbors=8
        )


if __name__ == '__main__':
    profile_select_directional()
