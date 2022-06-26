import unittest

import numpy as np

from pyinterpolate.distance.distance import calc_block_to_block_distance
from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.processing.polygon.structure import get_polyset_from_file, get_block_centroids_from_polyset
from pyinterpolate.variogram import build_experimental_variogram, TheoreticalVariogram
from pyinterpolate.variogram.regularization.block.inblock_semivariance import calculate_inblock_semivariance
from pyinterpolate.variogram.regularization.block.avg_block_to_block_semivariance import calculate_average_semivariance


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


class TestCalculateAverageSemivariance(unittest.TestCase):

    def test_avg_from_inblock(self):
        # Variogram model
        bc = get_block_centroids_from_polyset(AREAL_INPUT)
        experimental_variogram_of_areal_data = build_experimental_variogram(bc,
                                                                            step_size=STEP_SIZE,
                                                                            max_range=MAX_RANGE)
        theoretical_model = TheoreticalVariogram()
        theoretical_model.autofit(experimental_variogram_of_areal_data,
                                  number_of_ranges=64,
                                  number_of_sills=64,
                                  deviation_weighting='closest')

        # Inblock
        inblock_semivariances = calculate_inblock_semivariance(POINT_SUPPORT_INPUT['data'],
                                                               variogram_model=theoretical_model)

        # Distances
        distances_between_blocks = calc_block_to_block_distance(POINT_SUPPORT_INPUT['data'])

        # Avg semi
        avg_semivariance = calculate_average_semivariance(distances_between_blocks,
                                                          inblock_semivariances,
                                                          STEP_SIZE,
                                                          MAX_RANGE)

        self.assertIsInstance(avg_semivariance, np.ndarray)

        # TODO: add more tests
        # TODO: check SettingWithCopyWarning