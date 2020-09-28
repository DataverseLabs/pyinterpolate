import geopandas as gpd

from pyinterpolate.data_processing.data_transformation import get_centroids
from sample_data.data import Data


def test_get_centroids():
    data = Data()
    path_to_areal_file = data.poland_areas_dataset
    shapefile = gpd.read_file(path_to_areal_file)
    centroids = shapefile['geometry'].apply(get_centroids)
    c1x = 337895
    c1y = 321387

    assert ((type(centroids[0]) == tuple) and
            (int(centroids[1][0]) == c1x) and
            (int(centroids[1][1]) == c1y))


if __name__ == '__main__':
    test_get_centroids()
