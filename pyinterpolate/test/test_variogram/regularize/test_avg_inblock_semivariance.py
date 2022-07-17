import unittest
from typing import Tuple, Dict

import numpy as np

from pyinterpolate.distance.distance import calc_block_to_block_distance
from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.processing.polygon.structure import get_polyset_from_file
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


class TestCalculateAverageSemivariance(unittest.TestCase):

    def test_avg_from_inblock_real_world_data(self):
        # Variogram model
        bc = AREAL_INPUT['data'][:, 1:]
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

    def test_avg_from_inblock_artificial(self):

        sample_point_support, sample_b2b_distances = generate_test_blocks(100)

        # Inblock
        inblock_semivariances = calculate_inblock_semivariance(sample_point_support,
                                                               variogram_model=SAMPLE_VARIOGRAM)

        # Avg semi
        avg_semivariance = calculate_average_semivariance(sample_b2b_distances,
                                                          inblock_semivariances,
                                                          block_step_size=10,
                                                          block_max_range=80)

        lags = np.arange(10, 80, 10)

        for idx, lag in enumerate(lags):
            self.assertEqual(lag, avg_semivariance[idx, 0])

        mean_semi = float(np.mean(avg_semivariance[:, 1]))

        for row in avg_semivariance:
            self.assertAlmostEqual(mean_semi, row[1], places=1)
