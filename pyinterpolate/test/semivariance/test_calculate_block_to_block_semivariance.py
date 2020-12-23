import unittest
import os
import numpy as np
import geopandas as gpd
from pyinterpolate.io_ops.get_points_within_area import get_points_within_area
from pyinterpolate.io_ops.prepare_areal_shapefile import prepare_areal_shapefile
from pyinterpolate.semivariance.areal_semivariance.areal_semivariance import ArealSemivariance

from pyinterpolate.semivariance.areal_semivariance.block_to_block_semivariance.calculate_block_to_block_semivariance\
    import calculate_block_to_block_semivariance


class TestCalculateBlock2BlockSemivariance(unittest.TestCase):

    def test_calculate_block_to_block_semivariance(self):
        # Data prepration
        my_dir = os.path.dirname(__file__)

        areal_dataset = os.path.join(my_dir, '../sample_data/test_areas_pyinterpolate.shp')
        subset = os.path.join(my_dir, '../sample_data/test_points_pyinterpolate.shp')

        a_id = 'id'
        areal_val = 'value'
        points_val = 'value'

        # Get maximum range and set step size

        gdf = gpd.read_file(areal_dataset)

        total_bounds = gdf.geometry.total_bounds
        total_bounds_x = np.abs(total_bounds[2] - total_bounds[0])
        total_bounds_y = np.abs(total_bounds[3] - total_bounds[1])

        max_range = min(total_bounds_x, total_bounds_y)
        step_size = max_range / 4

        areal_data_prepared = prepare_areal_shapefile(areal_dataset, a_id, areal_val)
        points_in_area = get_points_within_area(areal_dataset, subset, areal_id_col_name=a_id,
                                                points_val_col_name=points_val)

        # Set areal semivariance class
        areal_semivariance = ArealSemivariance(areal_data_prepared, step_size, max_range, points_in_area)
        areal_semivariance.regularize_semivariogram()
        blocks = calculate_block_to_block_semivariance(areal_semivariance.within_area_points,
                                                       areal_semivariance.distances_between_blocks,
                                                       areal_semivariance.theoretical_semivariance_model)
        test_block = np.array(blocks[0][0])
        mean_semivariance = int(np.mean(test_block[:, 1]))
        self.assertEqual(mean_semivariance, 212, "Average semivariance should be 212 (decimal part)")


if __name__ == '__main__':
    unittest.main()
