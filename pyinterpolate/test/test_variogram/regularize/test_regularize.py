import os
import unittest

import numpy as np

from pyinterpolate.processing.structure import get_polyset_from_file
from pyinterpolate.variogram.regularization.regularize import Deconvolution

SHAPEFILE = '../../samples/areal_data/test_areas_pyinterpolate.shp'
AREAL_DATA = get_polyset_from_file(SHAPEFILE, value_col='value', index_col='idx')


class TestDeconvolution(unittest.TestCase):

    def test_fit_method(self):
        regularized_variogram = Deconvolution()
        output = regularized_variogram.fit(
            AREAL_DATA, np.zeros(10), 1, 3
        )
        print(output)

        self.assertTrue(False)
