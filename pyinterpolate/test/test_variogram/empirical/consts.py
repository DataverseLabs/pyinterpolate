import os
from dataclasses import dataclass
import numpy as np


def get_armstrong_data():
    my_dir = os.path.dirname(__file__)
    filename = 'armstrong_data.npy'
    filepath = f'../../samples/point_data/numpy/{filename}'
    path_to_the_data = os.path.join(my_dir, filepath)
    arr = np.load(path_to_the_data)
    return arr

"""
- - - - GENERAL - - - -
"""
@dataclass
class EmpiricalVariogramTestData:
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

"""
- - - - CALCULATE SEMIVARIANCE - - - -
"""
@dataclass
class EmpiricalSemivarianceData:
    output_we_omni = np.array([
        [1, 4.625, 24],
        [2, 5.227, 22],
        [3, 6.0, 20],
        [4, 4.444, 18],
        [5, 3.125, 16]
    ])

    output_armstrong_we_lag1 = 6.41
    output_armstrong_ns_lag1 = 4.98
    output_armstrong_ne_sw_lag2 = 7.459
    output_armstrong_nw_se_lag2 = 7.806
    output_armstrong_lag1 = 5.69

    output_weighted = np.array([
        [1, 30651.4, 48],
        [2, 40098.6, 68]
    ])

    directional_output_weighted = np.array([
        [2, 34480.6, 18],
        [4, 16409.3, 8],
        [6, 4166.9, 2]
    ])

"""
- - - - CALCULATE COVARIANCE - - - -
"""
@dataclass
class EmpiricalCovarianceData:
    # EXPECTED OUTPUTS
    output_variance = 4.248

    output_we_omni = np.array([
        [1, -0.543, 24],
        [2, -0.795, 22],
        [3, -1.26, 20],
        [4, -0.197, 18],
        [5, 1.234, 16]
    ])

    output_armstrong_we_lag1 = 4.643
    output_armstrong_ns_lag1 = 9.589
    output_armstrong_ne_sw_lag2 = 4.551
    output_armstrong_nw_se_lag2 = 6.331
    output_armstrong_omni_lag1 = 6.649

"""
- - - - CALCULATE EMPIRICAL VARIOGRAM CLASS - - - -
"""

@dataclass
class EmpiricalVariogramClassData:
    input_bounded = np.array([
        [0, 0, 2],
        [1, 0, 4],
        [2, 0, 6],
        [3, 0, 8],
        [4, 0, 10],
        [5, 0, 12],
        [6, 0, 14],
        [7, 0, 12],
        [8, 0, 10],
        [9, 0, 8],
        [10, 0, 6],
        [11, 0, 4],
        [12, 0, 2]
    ])

    param_step_size = 1
    param_max_range = 4
    param_step_size_bounded = 4
    param_bounded_max_range = (len(input_bounded) / 2) - 1
