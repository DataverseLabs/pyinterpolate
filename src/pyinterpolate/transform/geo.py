from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon


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

    lon = points.apply(lambda pt: pt.x)
    lat = points.apply(lambda pt: pt.y)
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


