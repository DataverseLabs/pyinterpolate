import unittest
import numpy as np

from pyinterpolate.processing.select_values import select_centroid_poisson_kriging_data, \
    select_neighbors_pk_centroid_with_angle

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
    [[0, 1, 1, 2, np.nan, 20],
     [1, 0, 2, 2, np.nan, 19]]
)

EXPECTED_RESULTS_NON_WEIGHTED = np.array(
    [[0, 1, 1, 2, np.nan, 15],
     [1, 0, 2, 2, np.nan, 20]]
)

# Angles
INDEXES = np.array(['a', 'b', 'c', 'd'])
KRIGING_DATA = np.array(([
    [0, 0, 10, 2, 0, 0],
    [0, 1, 20, 2, 5, 0],
    [1, 0, 30, 1, 15, 0],
    [-1, 0, 40, 4, 60, 0],
]))
EXPECTED_INDEXES = np.array(['a', 'b'])
EXPECTED_ROWS = np.array([
    [0, 0, 10, 2, 0, 0],
    [0, 1, 20, 2, 5, 0]
])


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
        array_equal = np.allclose(prepared, EXPECTED_RESULTS_WEIGHTED, rtol=1, atol=1, equal_nan=True)
        self.assertTrue(array_equal)

    def test_case_non_weighted(self):
        prepared = select_centroid_poisson_kriging_data(u_block_centroid=U_PT,
                                                        u_point_support=U_PS,
                                                        k_blocks=AREAL_INPUT,
                                                        k_point_support_dict=POINT_SUPPORT_INPUT,
                                                        nn=NN,
                                                        weighted=False,
                                                        max_range=2)
        array_equal = np.allclose(prepared, EXPECTED_RESULTS_NON_WEIGHTED, rtol=1, atol=1, equal_nan=True)
        self.assertTrue(array_equal)

    def test_case_with_angles(self):
        selected_indexes, selected_rows = select_neighbors_pk_centroid_with_angle(
            indexes=INDEXES,
            kriging_data=KRIGING_DATA,
            max_range=2,
            min_number_of_neighbors=2,
            use_all_neighbors_in_range=False
        )

        self.assertTrue(set(selected_indexes) == set(EXPECTED_INDEXES))

        array_equality_test = np.array_equal(selected_rows, EXPECTED_ROWS)
        self.assertTrue(array_equality_test)
