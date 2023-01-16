import unittest

import numpy as np

from pyinterpolate.distance.distance import calc_block_to_block_distance
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.variogram import build_experimental_variogram, TheoreticalVariogram
from pyinterpolate.variogram.regularization.block.inblock_semivariance import calculate_inblock_semivariance
from pyinterpolate.variogram.regularization.block.avg_inblock_semivariances import calculate_average_semivariance

DATASET = 'samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
MAX_RANGE = 400000
STEP_SIZE = 40000

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

bc = AREAL_INPUT.data[[AREAL_INPUT.cx, AREAL_INPUT.cy, AREAL_INPUT.value_column_name]].values
experimental_variogram_of_areal_data = build_experimental_variogram(bc,
                                                                    step_size=STEP_SIZE,
                                                                    max_range=MAX_RANGE)

theoretical_model = TheoreticalVariogram()
theoretical_model.autofit(experimental_variogram_of_areal_data,
                          number_of_ranges=64,
                          number_of_sills=64,
                          deviation_weighting='closest')


class TestCalculateAverageSemivariance(unittest.TestCase):

    def test_avg_from_inblock_real_world_data(self):
        # Inblock
        inblock_semivariances = calculate_inblock_semivariance(POINT_SUPPORT_INPUT,
                                                               variogram_model=theoretical_model)

        # Distances

        cls = [POLYGON_ID, POINT_SUPPORT_INPUT.x_col, POINT_SUPPORT_INPUT.y_col, POINT_SUPPORT_INPUT.value_column]

        distances_between_blocks = calc_block_to_block_distance(
            POINT_SUPPORT_INPUT.point_support[cls].values
        )

        # Avg semi
        avg_semivariance = calculate_average_semivariance(distances_between_blocks,
                                                          inblock_semivariances,
                                                          STEP_SIZE,
                                                          MAX_RANGE)

        self.assertIsInstance(avg_semivariance, np.ndarray)
