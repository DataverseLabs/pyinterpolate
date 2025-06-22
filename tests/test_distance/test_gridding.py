import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pyinterpolate.distance.gridding import create_grid, points_to_grid_avg

POINTS = np.array([
    [0, 0, 3],
    [0, 1, 4],
    [1, 0, 8],
    [1, 1, 1],
    [3, 3, 4],
    [0, 7, -1]
])


def test_gridding():
    newgrid = create_grid(
        POINTS[:, :-1], 3, 'hex'
    )
    newgrid.index.name = 'abc'
    assert isinstance(newgrid, gpd.GeoDataFrame)

    df = gpd.GeoDataFrame(POINTS)
    df.columns = ['x', 'y', 'value']
    df['geometry'] = gpd.points_from_xy(df['x'], df['y'])
    df.set_geometry('geometry', inplace=True)

    gridded = points_to_grid_avg(
        points=df,
        grid=newgrid,
        fillna=0
    )

    assert isinstance(gridded, gpd.GeoDataFrame)
    # gridded.plot(column='value')
    # plt.show()