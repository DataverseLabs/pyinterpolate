import unittest
import os
import numpy as np
import geopandas as gpd
from pyinterpolate.kriging.areal_poisson_kriging.area_to_point.atp_poisson_kriging import AtPPoissonKriging
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_weighted_semivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram

from pyinterpolate.io_ops.prepare_areal_shapefile import prepare_areal_shapefile
from pyinterpolate.io_ops.get_points_within_area import get_points_within_area
from pyinterpolate.transform.set_areal_weights import set_areal_weights

class TestAreaToPointPoissonKriging(unittest.TestCase):
    def test_atp_pk(self):
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
        step_size = max_range / 10

        areal_data_prepared = prepare_areal_shapefile(areal_dataset, a_id, areal_val)
        points_in_area = get_points_within_area(areal_dataset, subset, areal_id_col_name=a_id,
                                                points_val_col_name=points_val)

        # Get one area as unknown
        unknown_area_id = [1]

        u_points = points_in_area[points_in_area[:, 0] == unknown_area_id][0]

        k_areas = areal_data_prepared[areal_data_prepared[:, 0] != unknown_area_id]
        k_points = points_in_area[points_in_area[:, 0] != unknown_area_id]

        # Semivariance deconvolution

        semivar_modeling_data = set_areal_weights(k_areas, k_points)
        smv_model = calculate_weighted_semivariance(semivar_modeling_data, step_size, max_range)

        semivariogram = TheoreticalSemivariogram(k_areas[:, 2:], smv_model)

        semivariogram.find_optimal_model()

        # Poisson Kriging

        search_radius = max_range / 2
        number_of_observations = 3

        pkc = AtPPoissonKriging(regularized_model=semivariogram,
                                known_areas=k_areas,
                                known_areas_points=k_points)
        d = pkc.predict(u_points, number_of_observations, search_radius)

        est_areal = np.sum(d[:, 0])
        self.assertEqual(int(est_areal), 126, "Int of first value should be equal to 126")


if __name__ == '__main__':
    unittest.main()
