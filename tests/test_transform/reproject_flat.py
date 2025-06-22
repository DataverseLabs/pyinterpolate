import numpy as np
import pandas as pd

from pyinterpolate import reproject_flat


def test_reproject_flat_array():
    flat_array = [
        [50.1, 19.2, 100],
        [-33.99, -90, 7]
    ]

    arr = np.array(flat_array)

    reproj = reproject_flat(
        ds=arr,
        in_crs=4326,
        out_crs=2180
    )

    assert isinstance(reproj, np.ndarray)
    assert reproj[0][0] != arr[0][0]
    assert reproj[0][1] != arr[0][1]
    assert reproj[1][0] != arr[1][0]
    assert reproj[1][1] != arr[1][1]
    assert reproj[0][2] == arr[0][2]
    assert reproj[1][2] == arr[1][2]


def test_reproject_flat_dataframe():
    flat_array = [
        [50.1, 19.2, 100],
        [-33.99, -90, 7]
    ]

    arr = np.array(flat_array)
    columns = ['x', 'y', 'z']
    df = pd.DataFrame(arr, columns=columns)
    reproj = reproject_flat(
        ds=df,
        in_crs=4326,
        out_crs=2180,
        lon_col='x',
        lat_col='y'
    )

    assert isinstance(reproj, pd.DataFrame)
    assert set(reproj.columns) == set(columns)
