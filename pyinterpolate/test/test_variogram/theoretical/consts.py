import os
from dataclasses import dataclass
import numpy as np


"""
- - - - GENERAL - - - -
"""


def get_armstrong_data():
    my_dir = os.path.dirname(__file__)
    filename = 'armstrong_data.npy'
    filepath = f'../../samples/point_data/numpy/{filename}'
    path_to_the_data = os.path.join(my_dir, filepath)
    arr = np.load(path_to_the_data)
    return arr


@dataclass
class TheoreticalVariogramTestData:
    input_data_we = np.array([
        [0, 0, 8],
        [1, 0, 6],
        [2, 0, 4],
        [3, 0, 3],
        [4, 0, 6],
        [5, 0, 5],
        [6, 0, 7],
        [7, 0, 2],
        [8, 0, 8],
        [9, 0, 9],
        [10, 0, 5],
        [11, 0, 6],
        [12, 0, 3]
    ])

    input_zeros = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, 2, 0],
        [3, 3, 0],
        [3, 1, 0],
        [3, 2, 0]
    ])

    input_weighted = np.array(
        [
            [
                [0, 0, 735],
                [0, 1, 45],
                [0, 2, 125],
                [0, 3, 167],
                [1, 0, 450],
                [1, 1, 337],
                [1, 2, 95],
                [1, 3, 245],
                [2, 0, 124],
                [2, 1, 430],
                [2, 2, 230],
                [2, 3, 460],
                [3, 0, 75],
                [3, 1, 20],
                [3, 2, 32],
                [3, 3, 20]
            ],
            [
                [0, 0, 2],
                [0, 1, 3],
                [0, 2, 2],
                [0, 3, 3],
                [1, 0, 1],
                [1, 1, 3],
                [1, 2, 3],
                [1, 3, 2],
                [2, 0, 1],
                [2, 1, 2],
                [2, 2, 3],
                [2, 3, 1],
                [3, 0, 2],
                [3, 1, 2],
                [3, 2, 2],
                [3, 3, 1]
            ]]
    )

    output_zeros = 0
    param_step_size = 1
    param_max_range = 6


@dataclass
class TheoreticalVariogramModelsData:

    lags = np.arange(0, 10)

    nugget0 = 0
    nugget1 = 1
    nugget_random = np.random.random()

    sill0 = 0
    sill1 = 1
    sill_random = np.random.random()

    rang0 = 0
    rang1 = 1
    rang5 = 5
    rang10 = 10
    rang_random = np.random.randint(0, 100)
