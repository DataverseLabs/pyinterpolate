from typing import Union, List

import geopandas as gpd
import numpy as np
import shapely.geometry


def _define_box_step(lonlimit, latlimit, no_of_steps):
    if lonlimit < latlimit:
        step = lonlimit / (no_of_steps - 1)
    else:
        step = latlimit / (no_of_steps - 1)
    return step


def _define_hex_step(lonlimit, latlimit, no_of_steps):
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

    max_lon : float

    min_lat : float

    max_lat : float

    step : float

    Returns
    -------
    grid : gpd.GeoSeries
        Grid polygons.

    References
    ----------
    [1] Izan Pérez Cosano (Github @eperezcosano), How to draw a hexagonal grid on HTML Canvas
    https://eperezcosano.github.io/hex-grid/
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

    max_lon : float

    min_lat : float

    max_lat : float

    step : float

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


def _gen_hexagon(x, y, r, a):
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
    Function creates grid based on a set of points.

    Parameters
    ----------
    ds : Union[np.ndarray, List, gpd.GeoSeries]
        Data to be transformed, point coordinates [x, y] <-> [lon, lat] or GeoSeries with Point geometry.

    min_number_of_cells : int
        Expected number of cells in the smaller dimension.

    grid_type : str, default = "square"
        Available types: ``box``, ``hex``.

    Returns
    -------
    grid : gpd.GeoSeries
        Empty grid that can be used to aggregate points. It is GeoSeries of Polygons (squares or hexes).
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
        raise KeyError('Unrecognized grid type, available types: "square" or "hex".')

    grid = gpd.GeoDataFrame(grid)
    grid.columns = ['geometry']
    grid.set_geometry('geometry', inplace=True)

    if isinstance(ds, gpd.GeoSeries):
        grid.crs = ds.crs

    return grid


def points_to_grid(points: gpd.GeoDataFrame,
                   grid: gpd.GeoDataFrame,
                   fillna=None):
    """
    Function aggregates points over a specified grid.

    Parameters
    ----------
    points : geopandas GeoDataFrame

    grid : geopandas GeoSeries

    fillna : Any, optional
        The value to fill NaN's, if None given then this step is skipped.

    Returns
    -------
    aggregated : geopandas GeoDataFrame
    """

    joined = points.sjoin(grid, how='left', predicate='within')
    joined.drop('geometry', axis=1, inplace=True)
    grouped = joined.groupby('index_right').mean()
    aggregated = grid.join(grouped)

    if fillna is not None:
        aggregated.fillna(fillna, inplace=True)

    return aggregated


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.use('TkAgg')  # !IMPORTANT
    POINTS = np.array([
        [0, 0, 3],
        [0, 1, 4],
        [1, 0, 8],
        [1, 1, 1],
        [3, 3, 4],
        [0, 7, -1]
    ])

    newgrid = create_grid(
        POINTS[:, :-1], 3, 'hex'
    )
    newgrid.index.name = 'abc'

    df = gpd.GeoDataFrame(POINTS)
    df.columns = ['x', 'y', 'value']
    df['geometry'] = gpd.points_from_xy(df['x'], df['y'])
    df.set_geometry('geometry', inplace=True)

    gridded = points_to_grid(
        points=df,
        grid=newgrid,
        fillna=0
    )
    gridded.plot(column='value')
    plt.show()
