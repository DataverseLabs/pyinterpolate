import numpy as np
import geopandas as gpd
from pyinterpolate.data_processing.data_preparation.prepare_areal_shapefile import prepare_areal_shapefile
from pyinterpolate.data_processing.data_preparation.get_points_within_area import get_points_within_area
from pyinterpolate.semivariance.semivariogram_deconvolution.regularize_semivariogram import RegularizedSemivariogram


def test_regularize_semivariogram():
    reg_mod = RegularizedSemivariogram()

    # Data prepration
    areal_dataset = 'sample_data/test_areas_pyinterpolate.shp'
    subset = 'sample_data/test_points_pyinterpolate.shp'

    areal_id = 'id'
    areal_val = 'value'
    points_val = 'value'

    # Get maximum range and set step size

    gdf = gpd.read_file(areal_dataset)

    total_bounds = gdf.geometry.total_bounds
    total_bounds_x = np.abs(total_bounds[2] - total_bounds[0])
    total_bounds_y = np.abs(total_bounds[3] - total_bounds[1])

    max_range = min(total_bounds_x, total_bounds_y)
    step_size = max_range / 4
    lags = np.arange(0, max_range, step_size * 2)

    maximum_point_range = max_range / 10
    step_size_points = step_size / 10
    lags_points = np.arange(0, maximum_point_range, step_size_points * 2)

    areal_data_prepared = prepare_areal_shapefile(areal_dataset, areal_id, areal_val)
    points_in_area = get_points_within_area(areal_dataset, subset, areal_id_col_name=areal_id,
                                            points_val_col_name=points_val)

    # Fit

    reg_mod.fit(areal_data_prepared, lags, step_size, points_in_area)

    # Transform
    reg_mod.transform()

    regularized_smv = np.array([[0, 0],
                                [1, 120]])
    test_output = (reg_mod.optimal_regularized_model).astype(np.int)

    assert (test_output == regularized_smv[:, 1]).all()


if __name__ == '__main__':
    test_regularize_semivariogram()
