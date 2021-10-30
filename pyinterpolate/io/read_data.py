import geopandas as gpd
import numpy as np
import pandas as pd

from geopandas import points_from_xy


def read_txt(
        path: str, lon_col_no=0, lat_col_no=1, val_col_no=2, delim=',', skip_header=True, epsg='4326', crs=None
) -> gpd.GeoDataFrame:
    """
    Function reads data from a text file. Provided data format should include: longitude (x), latitude (y), value.
        You should provide crs or epsg, if it's not provided then epsg:4326 is used as a default value
        (https://epsg.io/4326). Data read by a function is converted into GeoDataFrame.

    INPUT:

    :param path: (str) path to the file,
    :param lon_col_no: (int) position of longitude column, default=0,
    :param lat_col_no: (int) position of latitude column, default=1,
    :param val_col_no: (int) position of value column, default=2,
    :param delim: (str) delimiter which separates columns,
    :param skip_header: (bool) skip the first row of data,
    :param epsg: (str) optional; if not provided and crs is None then algorithm sets epsg:4326 as a default value,
    :param crs: (str) optional;

    OUTPUT:

    :returns: (GeoDataFrame)"""

    data_arr = np.loadtxt(path, delimiter=delim)

    if skip_header:
        data_arr = data_arr[1:, :]

    gdf = gpd.GeoDataFrame(data=data_arr)
    gdf['geometry'] = points_from_xy(gdf[lon_col_no], gdf[lat_col_no])
    gdf.set_geometry('geometry', inplace=True)

    if crs is None:
        gdf.set_crs(epsg=epsg, inplace=True)
    else:
        gdf.set_crs(crs=crs, inplace=True)

    gdf = gdf[['geometry', val_col_no]]
    gdf.columns = ['geometry', 'value']

    return gdf


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


def read_block(
        path: str,
        val_col_name: str,
        geometry_col_name='geometry',
        id_col_name=None,
        epsg=None,
        crs=None
) -> gpd.GeoDataFrame:
    """
    Function reads block data from files supported by fiona / geopandas. Value column name must be provided. If geometry
        column has different name than 'geometry' it must be provided too. Id column name is optional, if not given then
        GeoDataFrame index is treated as id column. Optional parameters are epsg and crs. If any is set then read data
        is reprojected into a specific crs/epsg. Function return GeoDataFrame with columns: id, value, geometry,
        centroid.

    INPUT:

    :param path: (str) path to the file,
    :param val_col_name: (str) name of the column with analyzed values,
    :param geometry_col_name: (str) default='geometry', name of the column with blocks - polygons,
    :param id_col_name: (str or None) default=None, name of the column with unique indexes of areas,
    :param epsg: (str) default=None, optional parameter; if provided then read GeoDataFrame is reprojected to it,
    :param crs: (str) default=None, optional parameter; if provided then read GeoDataFrame is reprojected to it.

    OUTPUT:

    :returns: (GeoDataFrame) columns=['id', 'geometry', 'value', 'centroid']"""

    ########## INPUT TESTS

    # Check if id column is given
    if id_col_name is not None:
        columns = [id_col_name, geometry_col_name, val_col_name]
    else:
        columns = [geometry_col_name, val_col_name]

    # Check if crs | epsg is set that minimum one of those values is None
    check_geometry_params = np.array([True if x is None else False for x in [crs, epsg]]).any()
    if check_geometry_params:
        pass
    else:
        raise TypeError(f'Only one value CRS or EPSG should be provided but both are given: crs {crs}; epsg: {epsg}')

    ########## FUNCTION'S BODY

    gdf = gpd.read_file(path)

    # Check if columns exist
    gdf_cols = gdf.columns
    for c in columns:
        if c not in gdf_cols:
            raise TypeError(f'Given column {c} is not present in a dataset. Available columns: {gdf_cols}')

    ndf = gdf[columns].copy()
    ndf.geometry = ndf[geometry_col_name]

    # Check crs and epsg
    if (crs is not None) or (epsg is not None):
        if ndf.crs is None:
            ndf.set_crs(crs=crs, epsg=epsg, inplace=True)
        else:
            ndf.to_crs(crs=crs, epsg=epsg, inplace=True)

    # Get centroids
    centroid_col_name = 'centroid'
    ndf[centroid_col_name] = ndf.centroid

    # Add id column if not given
    if id_col_name is None:
        id_col_name = 'id'
        ndf[id_col_name] = ndf.index

    # Set columns
    output = ndf[[id_col_name, geometry_col_name, val_col_name, centroid_col_name]]

    return output


if __name__ == '__main__':
    bblock = '../../sample_data/areal_data/cancer_data.shp'
    bdf = read_block(bblock, 'rate')
    print(bdf.head())
