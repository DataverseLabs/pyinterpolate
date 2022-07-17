import unittest

import numpy as np

from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.processing.polygon.structure import get_polyset_from_file
from pyinterpolate.variogram.regularization.aggregated import AggregatedVariogram, regularize


DATASET = 'samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
MAX_RANGE = 400000
STEP_SIZE = 40000

AREAL_INPUT = get_polyset_from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)
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


class TestAggregatedRegularization(unittest.TestCase):

    def test_aggregated_variogram_class(self):
        agg_var = AggregatedVariogram(
            AREAL_INPUT, STEP_SIZE, MAX_RANGE, POINT_SUPPORT_INPUT['data'], verbose=True
        )

        # Check if raise AttributeError without regularization
        self.assertRaises(AttributeError, agg_var.show_semivariograms)

        _ = agg_var.regularize()

        expected_lags = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)
        self.assertTrue(np.array_equal(expected_lags, agg_var.agg_lags))

        self.assertTrue(np.all(
            agg_var.regularized_variogram[:, 1] >= 0
        ))

    def test_regularize_fn(self):
        variogram = regularize(
            AREAL_INPUT, STEP_SIZE, MAX_RANGE, POINT_SUPPORT_INPUT['data'], verbose=True
        )

        expected_lags = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)
        self.assertTrue(np.array_equal(expected_lags, variogram[:, 0]))

        self.assertTrue(np.all(
            variogram[:, 1] >= 0
        ))

    def test_compare_cls_to_fn(self):
        agg_var = AggregatedVariogram(
            AREAL_INPUT, STEP_SIZE, MAX_RANGE, POINT_SUPPORT_INPUT['data'], verbose=True
        )
        cls_variogram = agg_var.regularize()
        fn_variogram = regularize(
            AREAL_INPUT, STEP_SIZE, MAX_RANGE, POINT_SUPPORT_INPUT['data'], verbose=True
        )
        self.assertTrue(
            np.array_equal(
                cls_variogram, fn_variogram
            )
        )
