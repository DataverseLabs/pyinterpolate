import unittest
import numpy as np

from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.processing.polygon.structure import get_polyset_from_file
from pyinterpolate.variogram.regularization.deconvolution import Deconvolution

DATASET = '../../samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
MAX_RANGE = 400000
STEP_SIZE = 40000

AREAL_INPUT = get_polyset_from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID,
                                    layer_name=POLYGON_LAYER)
POINT_SUPPORT_INPUT = get_point_support_from_files(point_support_data_file=DATASET,
                                                   polygon_file=DATASET,
                                                   point_support_geometry_col=GEOMETRY_COL,
                                                   point_support_val_col=POP10,
                                                   polygon_geometry_col=GEOMETRY_COL,
                                                   polygon_index_col=POLYGON_ID,
                                                   use_point_support_crs=True,
                                                   dropna=True,
                                                   point_support_layer_name=POPULATION_LAYER,
                                                   polygon_layer_name=POLYGON_LAYER)



class TestDeconvolution(unittest.TestCase):

    def test_fit_method(self):
        dcv = Deconvolution(verbose=True)
        dcv.fit(agg_dataset=AREAL_INPUT,
                point_support_dataset=POINT_SUPPORT_INPUT['data'],
                agg_step_size=STEP_SIZE,
                agg_max_range=MAX_RANGE)

        fitted = dcv.initial_regularized_model
        initial_deviation = dcv.initial_deviation

        self.assertTrue(fitted is not None)

        expected_deviation = 0.1785
        self.assertAlmostEqual(initial_deviation, expected_deviation, 4)
