import unittest

from pyinterpolate.variogram.theoretical.spatial_dependency_index import calculate_spatial_dependence_index


class TestSpatialDependenceIndex(unittest.TestCase):

    def test_1(self):
        nugget = 10
        sill = 100
        eratio = (nugget / sill) * 100
        ename = 'strong'

        ratio, name = calculate_spatial_dependence_index(nugget, sill)
        self.assertEqual(eratio, ratio)
        self.assertEqual(ename, name)

    def test_raises_value_error(self):
        nugget = 0
        sill = 100

        self.assertRaises(
            ValueError, calculate_spatial_dependence_index, nugget, sill
        )
