import unittest

import numpy as np

from pyinterpolate.pipelines.multi_kriging import BlockToBlockKrigingComparison
from pyinterpolate.processing.preprocessing.blocks import PointSupport, Blocks
from pyinterpolate.variogram import TheoreticalVariogram


DATASET = 'samples/regularization/cancer_data.gpkg'
VARIOGRAM_MODEL_FILE = 'samples/regularization/regularized_variogram.json'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 8

AREAL_INPUT = Blocks()
AREAL_INPUT.from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)
POINT_SUPPORT_INPUT = PointSupport()
POINT_SUPPORT_INPUT.from_files(point_support_data_file=DATASET,
                               blocks_file=DATASET,
                               point_support_geometry_col=GEOMETRY_COL,
                               point_support_val_col=POP10,
                               blocks_geometry_col=GEOMETRY_COL,
                               blocks_index_col=POLYGON_ID,
                               use_point_support_crs=True,
                               point_support_layer_name=POPULATION_LAYER,
                               blocks_layer_name=POLYGON_LAYER)

THEORETICAL_VARIOGRAM = TheoreticalVariogram()
THEORETICAL_VARIOGRAM.from_json(VARIOGRAM_MODEL_FILE)


class TestBlockToBlockKrigingComparison(unittest.TestCase):

    def test_real_data(self):

        output_results = BlockToBlockKrigingComparison(variogram=THEORETICAL_VARIOGRAM,
                                                       blocks=AREAL_INPUT,
                                                       point_support=POINT_SUPPORT_INPUT,
                                                       no_of_neighbors=NN,
                                                       neighbors_range=None,
                                                       simple_kriging_mean=130,
                                                       training_set_frac=0.6,
                                                       raise_when_negative_error=False,
                                                       raise_when_negative_prediction=False,
                                                       allow_approx_solutions=True,
                                                       iters=2)
        for value in output_results.results.values():
            self.assertTrue(np.isnan(value))

        test_results = output_results.run_tests()
        test_nans = [np.isnan(value) for value in test_results.values()]
        if not all(test_nans):
            self.assertTrue(1)
        else:
            self.assertTrue(any(test_nans))

    def test_artificial_data(self):
        known_blocks = np.array([
            [1.0, 1, 1, 100],
            [2.0, 0, 1, 100],
            [3.0, 1, 0, 200],
            [4.0, 5, 1, 500],
            [5.0, 4, 2, 800]
        ])

        ps = {
            1.0: np.array([
                [0.9, 1.1, 1000],
                [1.1, 0.9, 2000],
                [0.8, 1.2, 1000]
            ]),
            2.0: np.array([
                [-0.1, 1.1, 300],
                [0.1, 1, 400]
            ]),
            3.0: np.array([
                [0.9, -0.2, 100],
                [1.1, -0.2, 200],
                [1.1, 0.2, 400],
                [0.9, 0.2, 200]
            ]),
            4.0: np.array([
                [4.9, 0.9, 200],
                [4.9, 1.1, 1000],
                [5.1, 0.9, 8000]
            ]),
            5.0: np.array([
                [3.8, 2.3, 600],
                [4.2, 1.7, 1000]
            ])
        }

        output_results = BlockToBlockKrigingComparison(variogram=THEORETICAL_VARIOGRAM,
                                                       blocks=known_blocks,
                                                       point_support=ps,
                                                       no_of_neighbors=2,
                                                       neighbors_range=None,
                                                       simple_kriging_mean=400,
                                                       training_set_frac=0.4,
                                                       raise_when_negative_error=True,
                                                       raise_when_negative_prediction=True,
                                                       iters=20)

        self.assertRaises(ValueError, output_results.run_tests)
