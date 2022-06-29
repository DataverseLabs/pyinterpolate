import unittest

import numpy as np

from pyinterpolate.distance.distance import calc_block_to_block_distance
from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.processing.polygon.structure import get_polyset_from_file, get_block_centroids_from_polyset
from pyinterpolate.variogram import TheoreticalVariogram, build_experimental_variogram
from pyinterpolate.variogram.regularization.block.avg_block_to_block_semivariances import \
    average_block_to_block_semivariances
from pyinterpolate.variogram.regularization.block.block_to_block_semivariance import \
    calculate_block_to_block_semivariance

DATASET = '../../samples/regularization/cancer_data.gpkg'
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


class TestAverageBlockToBlockSemivariance(unittest.TestCase):

    def test_real_world_data(self):
        # Get variogram model
        bc = get_block_centroids_from_polyset(AREAL_INPUT)
        experimental_variogram_of_areal_data = build_experimental_variogram(bc,
                                                                            step_size=STEP_SIZE,
                                                                            max_range=MAX_RANGE)
        theoretical_model = TheoreticalVariogram()
        theoretical_model.autofit(experimental_variogram_of_areal_data,
                                  number_of_ranges=64,
                                  number_of_sills=64,
                                  deviation_weighting='closest')

        # Calc block to block distances
        b2b_distances = calc_block_to_block_distance(POINT_SUPPORT_INPUT['data'])

        # Calc block to block
        b_semivars = calculate_block_to_block_semivariance(
            point_support=POINT_SUPPORT_INPUT['data'],
            block_to_block_distances=b2b_distances,
            semivariogram_model=theoretical_model
        )

        # Calc avg
        # TODO: remove fromiter
        b_arr = np.array(list(b_semivars.values()), dtype=float)
        lags = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)

        avg_semi = average_block_to_block_semivariances(b_arr, lags, STEP_SIZE)
        self.assertEqual(set(lags), set(avg_semi[:, 0]))
        self.assertEqual(int(avg_semi[0, 1]), 62)
        self.assertEqual(int(avg_semi[-1, 1]), 154)

        pts_per_lag = [416, 1716, 2358, 2892, 3342, 3780, 3964, 3832, 3696]
        self.assertEqual(set(pts_per_lag), set(avg_semi[:, -1]))
