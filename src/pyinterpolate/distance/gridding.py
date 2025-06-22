from typing import Union, List

import geopandas as gpd
import numpy as np
import shapely.geometry


def _define_box_step(lonlimit: float,
                     latlimit: float,
                     no_of_steps: int):
    """
    Function defines square grid size using difference between
    maximum and minimum values of longitude and latitude. Function chooses
    the smaller of the two differences and divides it by the number of steps
    minus 1.

    Parameters
    ----------
    lonlimit : float
        Difference between maximum and minimum longitude values.
        Must be positive.

    latlimit : float
        Difference between maximum and minimum latitude values.
        Must be positive.

    no_of_steps : int
        How many squares should fill the space between smaller of the two
        distances (lonlimit or latlimit)? Must be greater than 2.

    Returns
    -------

    """

    if no_of_steps < 2:
        raise ValueError(
            'Number of steps must be greater than 2.'
        )

    if lonlimit < latlimit:
        step = lonlimit / (no_of_steps - 1)
    else:
        step = latlimit / (no_of_steps - 1)
    return step


def _define_hex_step(lonlimit: float,
                     latlimit: float,
                     no_of_steps: float):
    """
    Function defines hexagonal grid size using difference between maximum
    and minimum values of longitude and latitude. Function chooses the smaller
    of the two differences and divides it by the number of steps minus 2.

    Parameters
    ----------
    lonlimit : float
        Difference between maximum and minimum longitude values.
        Must be positive.

    latlimit : float
        Difference between maximum and minimum latitude values.
        Must be positive.

    no_of_steps : int
        How many hexes should fill the space between smaller of the two
        distances (lonlimit or latlimit)? Must be greater than 3.

    Returns
    -------

    """

    if no_of_steps < 3:
        raise ValueError(
            'Number of steps must be greater than 3.'
        )

    if lonlimit < latlimit:
        step = lonlimit / (no_of_steps - 2)
    else:
        step = latlimit / (no_of_steps - 2)
    return step


def gen_hex_cells(min_lon, max_lon, min_lat, max_lat, step):
    """
    Function creates hexagonal grid.

    Parameters
    ----------
    min_lon : float
        Minimal longitude.

    max_lon : float
        Maximum longitude.

    min_lat : float
        Minimal latitude.

    max_lat : float
        Maximum latitude.

    step : float
        Parameter defining the size of the hexagon (two times distance from
        the center to the vertex).

    Returns
    -------
    grid : gpd.GeoSeries
        Grid polygons.

    References
    ----------
    [1] Izan Pérez Cosano (Github @eperezcosano), How to draw a hexagonal
    grid on HTML Canvas: https://eperezcosano.github.io/hex-grid/
    """
    grid = []

    r = step / 2
    a = np.pi / 3

    curr_y = min_lat
    while curr_y < (max_lat + step):
        xdirection = 1
        curr_x = min_lon
        curr_py = curr_y
        while curr_x < (max_lon + step):
            points = _gen_hexagon(curr_x, curr_py, r, a)

            grid.append(
                shapely.geometry.Polygon(points)
            )

            if xdirection == 1:
                curr_x = curr_x + r * (1 + np.cos(a))
                curr_py = curr_py + r * np.sin(a)
                xdirection = -1
            else:
                curr_x = curr_x + r * (1 + np.cos(a))
                curr_py = curr_py - r * np.sin(a)
                xdirection = 1

        curr_y = curr_y + step * np.sin(a)

    grid = gpd.GeoSeries(grid)
    return grid


def gen_square_cells(min_lon, max_lon, min_lat, max_lat, step):
    """
    Function creates box-polygons grid.

    Parameters
    ----------
    min_lon : float
        Minimal longitude.

    max_lon : float
        Maximum longitude.

    min_lat : float
        Minimal latitude.

    max_lat : float
        Maximum latitude.

    step : float
        Length of a side of the square.

    Returns
    -------
    grid : gpd.GeoSeries
        Grid polygons.
    """
    grid = []
    for llon_0 in np.arange(min_lon - 0.5 * step, max_lon, step):
        for llat_0 in np.arange(min_lat - 0.5 * step, max_lat, step):
            llon_1 = llon_0 + step
            llat_1 = llat_0 + step
            grid.append(
                shapely.geometry.box(llon_0, llat_0, llon_1, llat_1)
            )

    grid = gpd.GeoSeries(grid)
    return grid


def _gen_hexagon(x: float, y: float, r: float, a: float) -> list:
    """
    Function generates hexagon points (six vertices) based on the center and
    the radius of the hexagon.

    Parameters
    ----------
    x : float
        Parameter defing the center of the hexagon.

    y : float
        Parameter defing the center of the hexagon.

    r : float
        Parameter defining the radius of the hexagon.

    a : float
        Parameter defining the angle between the center and the vertex.

    Returns
    -------
    : list
        Hexagon vertices.
    """
    points = []

    for i in range(6):
        point = [
            x + r * np.cos(i * a),
            y + r * np.sin(i * a)
        ]
        points.append(point)

    return points


def create_grid(ds: Union[np.ndarray, List, gpd.GeoSeries],
                min_number_of_cells: int,
                grid_type='box'):
    """
    Function creates grid based on a set of coordinates.

    Parameters
    ----------
    ds : Union[np.ndarray, List, gpd.GeoSeries]
        Data to be transformed, point coordinates [x, y] <-> [lon, lat] or
        GeoSeries with Point geometry.

    min_number_of_cells : int
        Expected number of cells in the smaller dimension.

    grid_type : str, default = "box"
        Available types: ``box``, ``hex``.

    Returns
    -------
    grid : gpd.GeoSeries
        Empty grid that can be used to aggregate points. It is GeoSeries of
        Polygons (squares or hexes).

    Raises
    ------
    KeyError
        If grid type is not recognized.
    """

    if min_number_of_cells < 3:
        min_number_of_cells = 3

    if isinstance(ds, List):
        ds = np.ndarray(ds)

    # Define thresholds
    if isinstance(ds, np.ndarray):
        min_lon = min(ds[:, 0])
        max_lon = max(ds[:, 0])
        min_lat = min(ds[:, 1])
        max_lat = max(ds[:, 1])
    else:
        min_lon, min_lat, max_lon, max_lat = ds.total_bounds

    # Define limits
    lon_abs = np.abs(max_lon - min_lon)
    lat_abs = np.abs(max_lat - min_lat)

    # Build cells
    if grid_type == 'box':
        step = _define_box_step(lon_abs, lat_abs, min_number_of_cells)
        grid = gen_square_cells(
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            step=step
        )
    elif grid_type == 'hex':
        step = _define_hex_step(lon_abs, lat_abs, min_number_of_cells)
        grid = gen_hex_cells(
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            step=step
        )
    else:
        raise KeyError('Unrecognized grid type, '
                       'available types: "box" or "hex".')

    grid = gpd.GeoDataFrame(grid)
    grid.columns = ['geometry']
    grid.set_geometry('geometry', inplace=True)

    if isinstance(ds, gpd.GeoSeries):
        grid.crs = ds.crs

    return grid


def points_to_grid_avg(points: gpd.GeoDataFrame,
                       grid: gpd.GeoDataFrame,
                       fillna=None):
    """
    Function aggregates points over a specified grid.

    Parameters
    ----------
    points : GeoDataFrame
        Points to be aggregated with their respective values.

    grid : GeoSeries
        Polygonal grid to which the points will be aggregated.

    fillna : Any, optional
        The value to fill NaN's, if None given then this step is skipped.

    Returns
    -------
    aggregated : GeoDataFrame
        Aggregated points over the grid (mean value).
    """

    joined = points.sjoin(grid.rename_axis("index_right"),
                          how='left',
                          predicate='within')
    joined.drop('geometry', axis=1, inplace=True)
    grouped = joined.groupby('index_right').mean()
    aggregated = grid.join(grouped)

    if fillna is not None:
        aggregated.fillna(fillna, inplace=True)

    return aggregated
