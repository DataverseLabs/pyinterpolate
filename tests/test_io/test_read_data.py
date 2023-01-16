import unittest
import os
import geopandas as gpd
import numpy as np

from pyinterpolate.io.read_data import read_block, read_csv, read_txt


class TestReadData(unittest.TestCase):

	def test_read_txt(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../samples/point_data/txt/pl_dem.txt')
		data = read_txt(path_to_the_data, skip_header=False)

		# Check if data type is GeoDataFrame
		check_frame_1 = isinstance(data, np.ndarray)
		self.assertTrue(check_frame_1, "Instance of a data type should be numpy array")

	def test_read_csv(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../samples/point_data/csv/pl_dem.csv')
		data = read_csv(path_to_the_data, val_col_name='dem', lat_col_name='latitude', lon_col_name='longitude')

		# Check if data type is GeoDataFrame
		check_frame_1 = isinstance(data, np.ndarray)
		self.assertTrue(check_frame_1, "Instance of a data type should be numpy array")

	@staticmethod
	def _check_lists_equality(l1, l2):
		t = len(l1) == len(l2) and sorted(l1) == sorted(l2)
		return t

	def test_read_block(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../samples/areal_data/test_areas_pyinterpolate.shp')

		# Test if error occurs when wrong column names are given
		with self.assertRaises(TypeError):
			read_block(path_to_the_data, 'non_existing_col_name')

		# Test if error occurs when crs AND epsg are given both
		with self.assertRaises(TypeError):
			read_block(path_to_the_data, 'value', epsg='2180', crs='Lambert-93')

		# Test if proper columns are returned
		id_col = 'idx'
		val_col = 'value'
		geom_col = 'geometry'
		cent_col = 'centroid'
		t = read_block(path_to_the_data, val_col, geometry_col_name=geom_col, id_col_name=id_col)
		correct_list = [id_col, val_col, geom_col, cent_col]
		tcols = t.columns
		self.assertTrue(self._check_lists_equality(correct_list, tcols), f'Returned columns are not as expected!')


if __name__ == '__main__':
	unittest.main()
