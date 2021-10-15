import geopandas as gpd
import numpy as np
import pandas as pd

from geopandas import points_from_xy


def read_txt(
        path: str, val_col_no=2, lat_col_no=0, lon_col_no=1, delim=',', skip_header=True, epsg='4326', crs=None
) -> gpd.GeoDataFrame:
    """
    Function reads data from a text file. Provided data format should include: latitude, longitude, value. You should
        provide crs or epsg, if it's not provided then epsg:4326 is used as a default value (https://epsg.io/4326).
        Data read by a function is converted into GeoSeries.

    INPUT:

    :param path: (str) path to the file,
    :param val_col_no: (int) position of value column,
    :param lat_col_no: (int) position of latitude column,
    :param lon_col_no: (int) position of longitude column,
    :param delim: (str) delimiter which separates columns,
    :param skip_header: (bool) skip the first row of data,
    :param epsg: (str) optional; if not provided and crs is None then algorithm sets epsg:4326 as a default value,
    :param crs: (str) optional;

    OUTPUT:

    :returns: (GeoDataFrame)"""

    data_arr = np.loadtxt(path, delimiter=delim)

    if skip_header:
        data_arr = data_arr[1:, :]

    columns = ['y', 'x', 'value']

    gdf = gpd.GeoDataFrame(data=data_arr, columns=['y', 'x', 'value'])
    gdf['geometry'] = points_from_xy(gdf['x'], gdf['y'])
    gdf.set_geometry('geometry', inplace=True)

    if crs is None:
        gdf.set_crs(epsg=epsg, inplace=True)
    else:
        gdf.set_crs(crs=crs, inplace=True)

    return gdf[['geometry', 'value']]


def read_csv(
        path: str,
        val_col_name: str,
        lat_col_name: str,
        lon_col_name: str,
        delim=',',
        epsg=4326,
        crs=None
) -> gpd.GeoDataFrame:
    """
    Function reads data from a text file. Provided data format should include: latitude, longitude, value. You should
        provide crs or epsg, if it's not provided then epsg:4326 is used as a default value (https://epsg.io/4326).
        Data read by a function is converted into GeoSeries.

    INPUT:

    :param path: (str) path to the file,
    :param delim: (str) delimiter which separates columns,
    :param val_col_name: (str) name of the header with values,
    :param lat_col_name: (str) name of the latitude column (usually it is 'y' or 'latitude'),
    :param lon_col_name: (str) name of the longitude column (usually it is 'x' or 'longitude'),
    :param epsg: (str) optional; if not provided and crs is None then algorithm sets epsg:4326 as a default value,
    :param crs: (str) optional;

    OUTPUT:

    :returns: (GeoDataFrame)"""

    df = pd.read_csv(
        path, sep=delim
    )

    gdf = gpd.GeoDataFrame(df[[val_col_name, lat_col_name, lon_col_name]])

    # Very low possibility but check if there is a geometry column in the data
    if 'geometry' in gdf.columns:
        geometry_col_name = '_geometry'
    else:
        geometry_col_name = 'geometry'

    gdf[geometry_col_name] = points_from_xy(gdf[lon_col_name], gdf[lat_col_name])
    gdf.set_geometry(geometry_col_name, inplace=True)

    if crs is None:
        gdf.set_crs(epsg=epsg, inplace=True)
    else:
        gdf.set_crs(crs=crs, inplace=True)

    gdf = gdf[[geometry_col_name, val_col_name]]
    gdf.columns = ['geometry', 'value']

    return gdf
