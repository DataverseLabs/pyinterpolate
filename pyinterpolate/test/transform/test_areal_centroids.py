import unittest
import os
import geopandas as gpd
from pyinterpolate.transform.get_areal_centroids import get_centroids


class TestGetCentroids(unittest.TestCase):

    def test_get_centroids(self):
        my_dir = os.path.dirname(__file__)
        data = os.path.join(my_dir, '../sample_data/test_areas_pyinterpolate.shp')
        gdf = gpd.read_file(data)

        polygon = gdf.iloc[0]
        polygon = polygon['geometry']

        centroid = get_centroids(polygon)

        # Test length
        self.assertEqual(len(centroid), 2, "Centroid doesn't have two coordinates. Calculation error.")

        # Test values
        self.assertEqual(int(centroid[0]), 17, "Centroid calculation was wrong. Should be 17.")
        self.assertEqual(int(centroid[1]), 52, "Centroid calculation was wrong. Should be 52.")

        # Test type
        self.assertTrue(isinstance(centroid, tuple))


if __name__ == '__main__':
    unittest.main()
