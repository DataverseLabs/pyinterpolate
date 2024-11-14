from typing import Tuple

import geopandas as gpd


def points_to_lon_lat(points: gpd.GeoSeries) -> Tuple:
    """
    Function transform GeoSeries to lon / lat series.
    Parameters
    ----------
    points : GeoSeries

    Returns
    -------
    : Tuple[Series, Series]
        Longitude, latitude (x, y)
    """

    lon = points.apply(lambda pt: pt.x)
    lat = points.apply(lambda pt: pt.y)
    return lon, lat
