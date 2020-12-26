import unittest
import os
import numpy as np
from pyinterpolate.io_ops import prepare_areal_shapefile


class TestPrepareArealShapefile(unittest.TestCase):

    def test_prepare_areal_shapefile(self):

        my_dir = os.path.dirname(__file__)
        path_to_areal_file = os.path.join(my_dir, '../sample_data/test_areas_pyinterpolate.shp')

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
        self.assertTrue(isnan_test, "NaN values in last column")

        # Read with id column
        dataset_with_id = prepare_areal_shapefile(path_to_areal_file, id_column_name='id', dropnans=False)

        # Tests:
        # Must have 5 columns
        test_cols_dataset = len(dataset_with_id[0]) == 5
        self.assertTrue(test_cols_dataset, "Dataset should have 5 columns")


if __name__ == '__main__':
    unittest.main()
