import numpy as np
import geopandas as gpd
import pyproj
from geopandas.tools import sjoin


def _check_columns(areal_dataframe, areal_id, points_val, points_dataframe):
    """
    Function checks if both dataframes has the same id and/or value columns to prevent program from errors.
    :param areal_dataframe: (GeoDataFrame),
    :param areal_id: (string) name of the areal id column name,
    :param points_val: (string) name of the points value column name,
    :param points_dataframe: (GeoDataFrame).
    
    :return areal_df, points_df: If areal_id column name is in points_dataframe then points_dataframe column
        name is changed with prefix pts_; If points_val column name is in areal_dataframe then areal_dataframe
        column name is changed with prefix a_. Otherwise function returns original dataframes.
    """
    
    areal_columns = areal_dataframe.columns
    point_columns = points_dataframe.columns
    
    if areal_id in point_columns:
        points_dataframe.drop(areal_id, axis=1, inplace=True)
        
    if points_val in areal_columns:
        areal_dataframe.drop(points_val, axis=1, inplace=True)
    
    return areal_dataframe, points_dataframe


def get_points_within_area(area_shapefile,
                           points_shapefile,
                           areal_id_col_name,
                           points_val_col_name,
                           dropna=True,
                           points_geometry_col_name='geometry',
                           nans_to_zero=True):
    """
    Function prepares points data for further processing.

    INPUT:

    :param area_shapefile: (string) areal data shapefile address,
    :param points_shapefile: (string) points data shapefile address,
    :param areal_id_col_name: (string) name of the column with id of areas,
    :param points_val_col_name: (string) name of the value column of each point,
    :param dropna: (bool) if True then rows with NaN are deleted (areas without any points),
    :param points_geometry_col_name: (string) default is 'geometry' as in GeoPandas GeoDataFrames,
    :param nans_to_zero: (bool) if True then all NaN values are casted to 0.

    OUTPUT:

    :return: output_points_within_area (numpy array) of area id and array with point coordinates and values
        [area_id, [point_position_x, point_position_y, value]]
    """
    output_points_within_area = []

    areal_data = gpd.read_file(area_shapefile)
    points_data = gpd.read_file(points_shapefile)

    # Test if both files have the same columns

    areal_data, points_data = _check_columns(areal_data, areal_id_col_name, points_val_col_name, points_data)

    # Test if areal data has the same projection as points data
    # Match centroid points with areas
    if not pyproj.Proj(areal_data.crs).is_exact_same(pyproj.Proj(points_data.crs)):
        points_data = points_data.to_crs(areal_data.crs)

    # Join datasets
    joined_population_points = sjoin(points_data, areal_data, how='left')

    # Drop NaN values
    if dropna:
        joined_population_points = joined_population_points.dropna(axis=0)

    # Get all areas ids
    if areal_id_col_name is None:
        areal_id_col_name = 'index_right'

    areal_ids = joined_population_points[areal_id_col_name].unique()

    # get coordinate x and coordinate y of points
    try:
        joined_population_points['x'] = joined_population_points[points_geometry_col_name].apply(
            lambda _: _.x)
        joined_population_points['y'] = joined_population_points[points_geometry_col_name].apply(
            lambda _: _.y)
    except AttributeError:
        joined_population_points['x'] = joined_population_points[points_geometry_col_name].apply(
            lambda _: _[0].x)
        joined_population_points['y'] = joined_population_points[points_geometry_col_name].apply(
            lambda _: _[0].y)

    # Set cols to hold
    cols_to_hold = ['x', 'y', points_val_col_name]

    # Get data
    for area_id in areal_ids:
        dataset = joined_population_points[joined_population_points[areal_id_col_name] == area_id]
        dataset = dataset[cols_to_hold]

        # Remove nans
        if nans_to_zero:
            dataset.fillna(0, inplace=True)

        dataset_numpy = dataset.values
        output_points_within_area.append([area_id, dataset_numpy])

    return np.array(output_points_within_area)
