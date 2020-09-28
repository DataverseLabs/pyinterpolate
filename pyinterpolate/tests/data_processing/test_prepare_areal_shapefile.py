import numpy as np

from pyinterpolate.data_processing.data_preparation.prepare_areal_shapefile import prepare_areal_shapefile
from sample_data.data import Data


def test_prepare_areal_shapefile():
    data = Data()
    path_to_areal_file = data.poland_areas_dataset

    # Read without id column and without value column
    minimal_dataset = prepare_areal_shapefile(path_to_areal_file)

    # Tests:
    # Second column values are only np.nan
    test_nan_value = np.all(np.isnan(minimal_dataset[:, -1].astype(np.float)))
    assert test_nan_value

    # Read with id column
    dataset_with_id = prepare_areal_shapefile(path_to_areal_file, id_column_name='IDx', dropnans=False)

    # Tests:
    # Must have 4 columns
    test_cols_dataset = len(dataset_with_id[0]) == 4
    assert test_cols_dataset

    # Read with id and value column
    dataset_with_values = prepare_areal_shapefile(path_to_areal_file,
                                                  id_column_name='IDx', value_coulmn_name='LB RATES 2')

    # Tests:
    # Third column must be a tuple of points
    test_centroid_col_type = (type(dataset_with_values[0][2]) == tuple)
    assert test_centroid_col_type

    # Value column must be different than nan
    test_value_column = np.any(np.isfinite(dataset_with_values[:, -1].astype(np.float)))
    assert test_value_column


if __name__ == '__main__':
    test_prepare_areal_shapefile()
