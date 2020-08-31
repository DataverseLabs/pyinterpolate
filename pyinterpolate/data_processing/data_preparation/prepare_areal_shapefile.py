import numpy as np
import geopandas as gpd

from pyinterpolate.data_processing.data_transformation.get_areal_centroids import get_centroids


def prepare_areal_shapefile(areal_file_address, id_column_name=None, value_coulmn_name=None,
                            geometry_column_name='geometry', dropnans=True):
    """
    Function prepares areal shapefile for processing and transforms it into numpy array. Function returns
    two lists.
    :param areal_file_address: (string) path to the shapefile with areal data,
    :param id_column_name: (string) id column name, if not provided then index column is treated as the id,
    :param value_coulmn_name: (string) value column name, if not provided then all values are set to nan,
    :param geometry_column_name: (string) default is 'geometry',
    :param dropnans: (bool) if true then rows with nans are dropped,
    :return: areal_array: (numpy array) [area_id, area_geometry, centroid coordinate x, centroid coordinate y, value]
    """

    shapefile = gpd.read_file(areal_file_address)
    cols_to_hold = list()

    # Prepare index column
    if id_column_name is None:
        shapefile['id'] = shapefile.index
        cols_to_hold.append('id')
    else:
        cols_to_hold.append(id_column_name)

    # Prepare geometry column
    cols_to_hold.append(geometry_column_name)

    # Prepare value column
    if value_coulmn_name is None:
        shapefile['vals'] = np.nan
        cols_to_hold.append('vals')
    else:
        cols_to_hold.append(value_coulmn_name)

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
