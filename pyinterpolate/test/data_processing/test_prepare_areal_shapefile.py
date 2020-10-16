import numpy as np
from pyinterpolate.data_processing.data_preparation.prepare_areal_shapefile import prepare_areal_shapefile



def test_prepare_areal_shapefile():

    path_to_areal_file = 'sample_data/test_areas_pyinterpolate.shp'

    # Read without id column and without value column
    try:
        minimal_dataset = prepare_areal_shapefile(path_to_areal_file)
    except TypeError:
        assert True
    else:
        assert False

    minimal_dataset = prepare_areal_shapefile(path_to_areal_file, value_column_name='value')

    # Tests:
    # Last column values are not nan
    isnan_test = ~np.any(np.isnan(minimal_dataset[:, -1].astype(np.float)))
    assert isnan_test

    # Read with id column
    dataset_with_id = prepare_areal_shapefile(path_to_areal_file, id_column_name='id', dropnans=False)

    # Tests:
    # Must have 5 columns
    test_cols_dataset = len(dataset_with_id[0]) == 5
    assert test_cols_dataset

    # Read with id and value column
    dataset_with_values = prepare_areal_shapefile(path_to_areal_file,
                                                  id_column_name='id',
                                                  value_column_name='value')


if __name__ == '__main__':
    test_prepare_areal_shapefile()
