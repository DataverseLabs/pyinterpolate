import unittest
from typing import Tuple, Dict

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


# Artificial data
def generate_test_blocks(number_of_blocks: int, points_per_block=100) -> Tuple[Dict, Dict]:
    """
    Function generates a sample point support dict and random distances between blocks Dict.

    Parameters
    ----------
    number_of_blocks : int

    points_per_block : int, default = 100

    Returns
    -------
    : Tuple[Dict, Dict]
        sample_point_support, sample_distances
    """

    blocks = {}

    # Create blocks
    for x in range(number_of_blocks):
        test_block = np.random.random(size=(points_per_block, 3)) * 1000
        sx = str(x)
        blocks[sx] = test_block

    # Create distances
    distances_matrix = np.zeros(shape=(number_of_blocks, number_of_blocks))

    distances = {}
    for x in range(number_of_blocks):
        number_of_distances = number_of_blocks - x
        generated_distances = np.random.randint(0, 100, size=number_of_distances)
        distances_matrix[x, x:] = generated_distances
        distances_matrix[x:, x] = generated_distances
        distances_matrix[x, x] = 0

    for idx, row in enumerate(distances_matrix):
        distances[str(idx)] = row

    return blocks, distances


SAMPLE_VARIOGRAM = TheoreticalVariogram(model_params={
    'nugget': 0,
    'sill': 95,
    'range': 50,
    'name': 'gaussian'
})


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
        b_arr = np.fromiter(b_semivars.values(), dtype=np.dtype((float, 3)))
        lags = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)

        avg_semi = average_block_to_block_semivariances(b_arr, lags, STEP_SIZE)
        print(avg_semi)
        self.assertTrue(1)