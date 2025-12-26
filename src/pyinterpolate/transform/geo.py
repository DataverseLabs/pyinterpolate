from typing import Tuple, Union
from numpy.typing import ArrayLike

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry


def geometry_and_values_array(geometry,
                              values) -> ArrayLike:
    """
    Function creates single object from geometries and aggregated values.

    Parameters
    ----------
    geometry : ArrayLike

    values : ArrayLike

    Returns
    -------
    : numpy array
    """

    if len(geometry) != len(values):
        raise ValueError(
            'Number of geometries must be equal to number of values'
        )

    arr = []

    if isinstance(geometry, pd.Series):
        geometry = geometry.values

    if isinstance(geometry, pd.DataFrame):
        geometry = geometry.values

    if isinstance(values, pd.Series):
        values = values.values

    is_point = isinstance(geometry[0], Point)

    if is_point:
        for idx, rec in enumerate(geometry):
            arr.append(
                [rec.x, rec.y, values[idx]]
            )
    else:
        greater_than_2_dims = len(geometry[0]) > 2
        if greater_than_2_dims:
            for idx, rec in enumerate(geometry):
                rlist = rec.tolist()
                rlist.append(values[idx])
                arr.append(rlist)
        else:
            for idx, rec in enumerate(geometry):
                arr.append(
                    [rec[0], rec[1], values[idx]]
                )

    return np.array(arr)


def largest_geometry(geometry: MultiPolygon) -> Polygon:
    """
    Samples largest polygon from multiple polygons.

    Parameters
    ----------
    geometry : MultiPolygon

    Returns
    -------
    : Polygon
    """

    areas = [p.area for p in geometry.geoms]

    idx = 0
    mx = areas[idx]
    for i in range(len(areas)):
        ar = areas[i]
        if ar > mx:
            mx = ar
            idx = i

    poly = geometry.geoms[idx]
    return poly


def join_any_geometry_and_values(geometry,
                                 values,
                                 geometry_column_name: str = 'geometry',
                                 values_column_name: str = 'values') -> gpd.GeoDataFrame:
    """
    Function creates single object from geometries and aggregated values.

    Parameters
    ----------
    geometry : ArrayLike

    values : ArrayLike

    geometry_column_name : str, default = 'geometry'

    values_column_name : str, default = 'value'

    Returns
    -------
    : gpd.GeoDataFrame
    """

    if len(geometry) != len(values):
        raise ValueError(
            'Number of geometries must be equal to number of values'
        )

    if isinstance(values, pd.Series):
        values = values.values
    elif isinstance(values, pd.DataFrame):
        val_column_name = values.columns[0]
        values = values[val_column_name].values

    if isinstance(geometry, pd.DataFrame):
        geom_column_name = geometry.columns[0]
        geometry = geometry[geom_column_name].values

    if isinstance(geometry, gpd.GeoDataFrame):
        gdf = geometry.copy(deep=True)
        gdf[values_column_name] = values
    else:
        g0 = geometry[0]
        if isinstance(g0, BaseGeometry):
            gdf = gpd.GeoDataFrame(
                values, columns=[values_column_name], geometry=geometry
            )
            gdf.columns = [values_column_name, geometry_column_name]
        else:
            raise TypeError('Passed geometry must be basic geometry types'
                            ' supported by shapely (Point, MultiPoint,'
                            'LineString, MultiLineString, Polygon, '
                            'MultiPolygon)')

    return gdf


def join_point_geometry_and_values(geometry,
                                   values,
                                   geometry_column_name: str = 'geometry',
                                   values_column_name: str = 'values') -> gpd.GeoDataFrame:
    """
    Function creates single object from geometries and aggregated values.

    Parameters
    ----------
    geometry : ArrayLike

    values : ArrayLike

    geometry_column_name : str, default = 'geometry'

    values_column_name : str, default = 'value'

    Returns
    -------
    : gpd.GeoDataFrame
    """

    if len(geometry) != len(values):
        raise ValueError(
            'Number of geometries must be equal to number of values'
        )

    if isinstance(values, pd.Series):
        values = values.values
    elif isinstance(values, pd.DataFrame):
        val_column_name = values.columns[0]
        values = values[val_column_name].values

    if isinstance(geometry, pd.DataFrame):
        geometry = geometry.values

    if isinstance(geometry, gpd.GeoDataFrame):
        gdf = geometry.copy(deep=True)
        gdf[values_column_name] = values
    else:
        points = [Point(p) for p in geometry]
        gdf = gpd.GeoDataFrame(
            values, columns=[values_column_name], geometry=points
        )
        gdf.columns = [values_column_name, geometry_column_name]

    return gdf


def points_to_lon_lat(points: gpd.GeoSeries) -> Tuple:
    """
    Function transform GeoSeries to lon / lat series.

    Parameters
    ----------
    points : GeoSeries
        Shapely points as GeoSeries.

    Returns
    -------
    : Tuple[Series, Series]
        Longitude, latitude (x, y)
    """

    lon = points.apply(lambda pt: pt.x if isinstance(pt, Point) else None)
    lat = points.apply(lambda pt: pt.y if isinstance(pt, Point) else None)
    return lon, lat


def reproject_flat(ds: Union[pd.DataFrame, np.ndarray],
                   in_crs,
                   out_crs,
                   lon_col=None,
                   lat_col=None) -> Union[pd.DataFrame, np.ndarray]:
    """
    Function reprojects geometries in pandas DataFrames and numpy arrays.

    Parameters
    ----------
    ds : DataFrame | array
        DataFrame with longitude (x) and latitude (y) columns or numpy
        array where first two columns are coordinates - longitude and latitude.

    in_crs : CRS
        Projection of input dataset.

    out_crs : CRS
        Projection of the output.

    lon_col : Hashable
        The name of longitude column.

    lat_col : Hashable
        The name of latitude column.

    Returns
    -------
    : DataFrame | array
        Returns the same data structure as was given in the input.
        DataFrame has the same columns, values in longitude
        and latitude columns are changed. The same for numpy array -
        only first two columns have changed values.
    """

    if isinstance(ds, pd.DataFrame):
        points = gpd.points_from_xy(
            x=ds[lon_col],
            y=ds[lat_col],
            crs=in_crs
        )
    else:
        points = gpd.points_from_xy(
            x=ds[:, 0],
            y=ds[:, 1],
            crs=in_crs
        )

    # Transform points
    points_repro = points.to_crs(out_crs)
    longitudes = [p.x for p in points_repro]
    latitudes = [p.y for p in points_repro]

    # return new output
    ds_t = ds.copy()

    if isinstance(ds, pd.DataFrame):
        ds_t.drop(columns=[lon_col, lat_col], inplace=True)
        ds_t[lon_col] = longitudes
        ds_t[lat_col] = latitudes
    else:
        ds_t[:, 0] = longitudes
        ds_t[:, 1] = latitudes

    return ds_t


if __name__ == '__main__':
    from shapely.geometry import Polygon
    pol1 = Polygon([[0, 0], [1, 1], [1, 0], [0, 0]])
    pol2 = Polygon([[1, 2], [4, 6], [2, 0], [1, 2]])

    mpol = MultiPolygon([pol1, pol2])

    assert largest_geometry(mpol) == pol2
