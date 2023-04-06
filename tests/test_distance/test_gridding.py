import unittest
import numpy as np
import geopandas as gpd
from pyinterpolate.distance.gridding import create_grid, points_to_grid


POINTS = np.array([
        [0, 0, 3],
        [0, 1, 4],
        [1, 0, 8],
        [1, 1, 1],
        [3, 3, 4],
        [0, 7, -1]
    ])

GEOPOINTS = gpd.GeoDataFrame(POINTS)
GEOPOINTS.columns = ['x', 'y', 'value']
GEOPOINTS['geometry'] = gpd.points_from_xy(GEOPOINTS['x'], GEOPOINTS['y'])
GEOPOINTS.geometry = GEOPOINTS['geometry']


class TestCreateGrid(unittest.TestCase):

    def test_1(self):
        no_of_blocks = np.arange(-2, 10)

        for blck in no_of_blocks:
            newgrid_hex = create_grid(
                POINTS[:, :-1], blck, 'hex'
            )
            newgrid_box = create_grid(POINTS[:, :-1], blck, 'box')
            self.assertIsInstance(newgrid_hex, gpd.GeoDataFrame)
            self.assertIsInstance(newgrid_box, gpd.GeoDataFrame)

    def test_2(self):
        blck = 1
        newgrid = create_grid(POINTS[:, :-1], blck, 'hex')
        self.assertEqual(newgrid.values.size, 12)

    def test_3(self):
        blck = 10
        self.assertRaises(KeyError, create_grid, POINTS[:, :-1], blck, 'triangle')


class TestPointsToGrid(unittest.TestCase):

    def test_1(self):
        no_of_blocks = np.arange(-2, 10)

        for blck in no_of_blocks:
            newgrid_hex = create_grid(
                POINTS[:, :-1], blck, 'hex'
            )
            newgrid_box = create_grid(POINTS[:, :-1], blck, 'box')

            gridded_hex = points_to_grid(GEOPOINTS, grid=newgrid_hex)
            gridded_box = points_to_grid(GEOPOINTS, grid=newgrid_box)

            self.assertIsInstance(gridded_hex, gpd.GeoDataFrame)
            self.assertIsInstance(gridded_box, gpd.GeoDataFrame)

    def test_2(self):
        blck = 10
        newgrid = create_grid(POINTS[:, :-1], blck, 'hex')
        gridded = points_to_grid(GEOPOINTS, newgrid)
        any_na = gridded['value'].isna().any()
        self.assertTrue(any_na)

    def test_3(self):
        blck = 10
        newgrid = create_grid(POINTS[:, :-1], blck, 'hex')
        gridded = points_to_grid(GEOPOINTS, newgrid, fillna=0)
        any_na = gridded['value'].isna().any()
        self.assertFalse(any_na)
