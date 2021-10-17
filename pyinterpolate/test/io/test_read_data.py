import unittest
import os
import geopandas as gpd
from pyinterpolate.io.read_data import read_csv, read_txt


class TestReadData(unittest.TestCase):

	def test_read_txt(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../samples/point_data/txt/pl_dem.txt')
		data = read_txt(path_to_the_data, skip_header=False)

		# Check if data type is GeoDataFrame
		check_frame_1 = isinstance(data, gpd.GeoDataFrame)
		self.assertTrue(check_frame_1, "Instance of a data type should be GeoDataFrame")

		# Check if lat is switched with lon
		base_x = data.iloc[0].geometry.x
		other_data = read_txt(path_to_the_data, lon_col_no=1, lat_col_no=0)
		other_y = other_data.iloc[0].geometry.y

		self.assertEqual(base_x, other_y, 'Something went wrong, geometries should be swapped')

	def test_read_csv(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../samples/point_data/csv/pl_dem.csv')
		data = read_csv(path_to_the_data, val_col_name='dem', lat_col_name='latitude', lon_col_name='longitude')

		# Check if data type is GeoDataFrame
		check_frame_1 = isinstance(data, gpd.GeoDataFrame)
		self.assertTrue(check_frame_1, "Instance of a data type should be GeoDataFrame")

		# Check if lat is switched with lon
		base_x = data.iloc[0].geometry.x
		other_data = read_csv(path_to_the_data, val_col_name='dem', lat_col_name='longitude', lon_col_name='latitude')
		other_y = other_data.iloc[0].geometry.y

		self.assertEqual(base_x, other_y, 'Something went wrong, geometries should be swapped')


if __name__ == '__main__':
	unittest.main()
