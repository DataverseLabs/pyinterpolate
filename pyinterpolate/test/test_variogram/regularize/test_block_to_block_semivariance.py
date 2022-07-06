import unittest
import numpy as np

from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram
from pyinterpolate.variogram.regularization.block.block_to_block_semivariance import block_pair_semivariance, \
    calculate_block_to_block_semivariance


TEST_BLOCK_A = np.random.random(size=(100, 3))
TEST_BLOCK_B = np.random.random(size=(50, 3))
TEST_BLOCK_XYZ = np.random.random(size=(25, 3))
SAMPLE_VARIOGRAM = TheoreticalVariogram(model_params={
    'nugget': 0,
    'sill': 0.95,
    'range': 0.5,
    'name': 'gaussian'
})

POINT_SUPPORT = {
    'area a': TEST_BLOCK_A,
    'area b': TEST_BLOCK_B,
    'area c': TEST_BLOCK_XYZ
}

BLOCK_TO_BLOCK_DISTANCES = {
    'area a': [0, 1, 2],
    'area b': [1, 0, 1],
    'area c': [2, 1, 0]
}



class TestBlockPairSemivariance(unittest.TestCase):

    def test_fn(self):
        semivars = block_pair_semivariance(TEST_BLOCK_A[:, :-1],
                                           TEST_BLOCK_B[:, :-1],
                                           semivariogram_model=SAMPLE_VARIOGRAM)
        self.assertIsInstance(semivars, float)
        self.assertTrue(semivars >= 0)


class TestCalculateCentroidBlock2BlockSemivariance(unittest.TestCase):
    """
    point_support : Dict
                    Point support dict in the form:

                    point_support = {
                          'area_id': [numpy array with points and their values]
                    }
    """

    def test_fn(self):
        semivars = calculate_block_to_block_semivariance(
            point_support=POINT_SUPPORT,
            block_to_block_distances=BLOCK_TO_BLOCK_DISTANCES,
            semivariogram_model=SAMPLE_VARIOGRAM
        )
        expected_keys = {('area a', 'area a'), ('area a', 'area b'), ('area a', 'area c'),
                         ('area b', 'area a'), ('area b', 'area b'), ('area b', 'area c'),
                         ('area c', 'area a'), ('area c', 'area b'), ('area c', 'area c')}
        self.assertEqual(set(semivars.keys()), expected_keys)

        expected_values = {
            ('area a', 'area a'): [0, 0, 0],
            ('area b', 'area b'): [0, 0, 0],
            ('area c', 'area c'): [0, 0, 0]
        }

        for k, item in expected_values.items():
            self.assertEqual(semivars[k], item)
