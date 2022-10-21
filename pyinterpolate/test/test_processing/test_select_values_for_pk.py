import unittest
import numpy as np

from pyinterpolate.processing.select_values import select_centroid_poisson_kriging_data

AREAL_INPUT = np.array(
        [[0, 0, 0, 0],
         [1, 0, 1, 1],
         [2, 1, 0, 2]]
    )

POINT_SUPPORT_INPUT = {
        0: np.array(
            [[0.1, 0.1, 5],
             [-0.1, -0.1, 10]]
        ),
        1: np.array(
            [[0.1, 1.1, 3],
             [0.1, 0.9, 3],
             [0.1, 0.5, 4],
             [0.1, 1.5, 10]]
        ),
        2: np.array(
            [[1.1, 0.1, 4],
             [0.9, 0.1, 15]]
        )
}

U_PT = np.array([3, 2, 2])

U_PS = np.array(
    [[1.9, 1.9, 5],
     [2.1, 2.1, 8],
     [1.5, 2.5, 10]]
)

NN = 2
WEIGHTED = True

EXPECTED_RESULTS_WEIGHTED = np.array(
    [[0, 1, 1, 2, 20],
     [1, 0, 2, 2, 19]]
)

EXPECTED_RESULTS_NON_WEIGHTED = np.array(
    [[0, 1, 1, 2, 15],
     [1, 0, 2, 2, 20]]
)


class TestSelectPoissonKrigingData(unittest.TestCase):

    def test_case_weighted(self):
        prepared = select_centroid_poisson_kriging_data(u_block_centroid=U_PT,
                                                        u_point_support=U_PS,
                                                        k_blocks=AREAL_INPUT,
                                                        k_point_support_dict=POINT_SUPPORT_INPUT,
                                                        nn=NN,
                                                        weighted=WEIGHTED,
                                                        max_range=2)

        self.assertIsInstance(prepared, np.ndarray)
        prepared_as_int = prepared.astype(int)
        array_equal = np.array_equal(prepared_as_int, EXPECTED_RESULTS_WEIGHTED)
        self.assertTrue(array_equal)

    def test_case_non_weighted(self):
        prepared = select_centroid_poisson_kriging_data(u_block_centroid=U_PT,
                                                        u_point_support=U_PS,
                                                        k_blocks=AREAL_INPUT,
                                                        k_point_support_dict=POINT_SUPPORT_INPUT,
                                                        nn=NN,
                                                        weighted=False,
                                                        max_range=2)
        prepared_as_int = prepared.astype(int)
        array_equal = np.array_equal(prepared_as_int, EXPECTED_RESULTS_NON_WEIGHTED)
        self.assertTrue(array_equal)
