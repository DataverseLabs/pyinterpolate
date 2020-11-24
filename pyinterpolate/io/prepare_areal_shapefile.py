import numpy as np
import geopandas as gpd

from pyinterpolate.transform.get_areal_centroids import get_centroids


def prepare_areal_shapefile(areal_file_address,
                            id_column_name=None,
                            value_column_name=None,
                            geometry_column_name='geometry',
                            dropnans=True):
    """Function prepares areal shapefile for processing and transforms it into numpy array. Function returns two lists.

    INPUT:

    :param areal_file_address: (string) path to the shapefile with area data,
    :param id_column_name: (string) id column name, if not provided then index column is treated as the id,
    :param value_column_name: (string) value column name, if not provided then all values are set to NaN,
    :param geometry_column_name: (string) default is 'geometry' as in GeoPandas GeoDataFrames,
    :param dropnans: (bool) if True then rows with NaN are dropped.

    OUTPUT:

    :return: areal_array (numpy array) of area id, area geometry, coordinate of centroid x, coordinate of centroid y,
        value:

        [area_id, area_geometry, centroid coordinate x, centroid coordinate y, value]
    """

    # Test if value column name is None and dropnans is True
    if (value_column_name is None) and dropnans:
        raise TypeError('You cannot leave value_column_name as None and set dropnans to True because function '
                        'will return empty list')

    shapefile = gpd.read_file(areal_file_address)
    cols_to_hold = list()

    # Prepare index column
    if id_column_name is None:
        shapefile['id_generated'] = shapefile.index
        cols_to_hold.append('id_generated')
    else:
        cols_to_hold.append(id_column_name)

    # Prepare geometry column
    cols_to_hold.append(geometry_column_name)

    # Prepare value column
    if value_column_name is None:
        shapefile['vals_generated'] = np.nan
        cols_to_hold.append('vals_generated')
    else:
        cols_to_hold.append(value_column_name)

    # Remove unwanted columns
    gdf = shapefile.copy()
    for col in gdf.columns:
        if col not in cols_to_hold:
            gdf.drop(labels=col, axis=1, inplace=True)

    # Set order of columns
    gdf = gdf[cols_to_hold]

    # Remove rows with nan's
    if dropnans:
        gdf.dropna(axis=0, inplace=True)

    # Extract values into numpy array
    areal_array = gdf.values

    # Get areal centroids
    centroids = [get_centroids(x) for x in areal_array[:, 1]]
    centroids = np.array(centroids)

    # Combine data into areal dataset
    areal_dataset = np.c_[areal_array[:, :2], centroids, areal_array[:, -1]]

    return areal_dataset
