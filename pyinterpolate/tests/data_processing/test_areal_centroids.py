import geopandas as gpd
from pyinterpolate.data_processing.data_transformation.get_areal_centroids import get_centroids


def test_get_centroids():

    gdf = gpd.read_file('sample_data/test_areas_pyinterpolate.shp')

    polygon = gdf.iloc[0]
    polygon = polygon['geometry']

    centroid = get_centroids(polygon)

    # Test length
    assert len(centroid) == 2

    # Test values
    assert int(centroid[0]) == 17
    assert int(centroid[1]) == 52

    # Test type
    assert type(centroid) == tuple


if __name__ == '__main__':
    test_get_centroids()
