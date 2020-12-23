import unittest
import os
from pyinterpolate.io_ops import get_points_within_area


class TestGetPoints(unittest.TestCase):

    def test_get_points(self):

        my_dir = os.path.dirname(__file__)
        areal_data = os.path.join(my_dir, '../sample_data/test_areas_pyinterpolate.shp')
        point_data = os.path.join(my_dir, '../sample_data/test_points_pyinterpolate.shp')

        points_val_col = 'value'
        areal_id_col = 'id'

        points_within_area = get_points_within_area(area_shapefile=areal_data,
                                                    points_shapefile=point_data,
                                                    areal_id_col_name=areal_id_col,
                                                    points_val_col_name=points_val_col,
                                                    dropna=True,
                                                    points_geometry_col_name='geometry',
                                                    nans_to_zero=True)

        self.assertEqual(len(points_within_area), 6, "Wrong number of points within area, should be 6")


if __name__ == '__main__':
    unittest.main()
