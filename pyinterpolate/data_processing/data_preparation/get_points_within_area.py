import numpy as np
import geopandas as gpd
import pyproj
from geopandas.tools import sjoin


def get_points_within_area(area_shapefile, points_shapefile, dropna=True,
                           areal_id_col_name=None, points_val_col_name=None,
                           points_geometry_col_name='geometry',
                           nans_to_zero=True):
    """
    Function prepares points data for further processing.
    :param area_shapefile: (string) areal data shapefile address,
    :param points_shapefile: (string) points data shapefile address,
    :param dropna: (bool) if True then rows with NaN are deleted (areas without any points).
    :param areal_id_col_name: (string) name of the column with id, if None then function uses index column,
    :param points_val_col_name: (string) name of the value column of each point, if None then first column other than
        points_geometry_col_name is used,
    :param points_geometry_col_name: (string) default 'geometry',
    :param nans_to_zero: (bool) if true then all nan value is casted to 0,
    :return: output_points_within_area (numpy array) [area_id, [point_position_x, point_position_y, value]]
    """
    output_points_within_area = []

    areal_data = gpd.read_file(area_shapefile)
    points_data = gpd.read_file(points_shapefile)

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
