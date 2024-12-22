"""
Core data structures for block interpolation.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
2. Taher Chegini | @cheginit

Changes in version 1.0
- Package doesn't read data files, it must be provided as GeoDataFrame
"""
from typing import Union, Hashable, Dict
from numpy.typing import ArrayLike

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.distance.angular import calc_angles
from pyinterpolate.distance.point import point_distance
from pyinterpolate.transform.geo import points_to_lon_lat


# TODO: if multipolygon then get coordinates / representative points from the largest block - as an option
class Blocks:
    """Class represents aggregated blocks data.

    Parameters
    ----------
    ds : gpd.GeoDataFrame
        Dataset with block values.

    value_column_name : Any
        Name of the column with block rates.

    geometry_column_name : Any, default = 'geometry'
        Name of the column with a block geometry.

    index_column_name : Any, optional
        Name of the indexing column.

    representative_points_column_name : Any, optional
        The column with representative points or coordinates.

    representative_points_from_centroid : bool, default = False
        Calculate representative points from block centroids.

    representative_points_from_random_sample : bool, default = False
        Calculate representative points from the point sampled from block geometry.

    distances_between_representative_points : bool, default = True
        Calculate distances between representative points during class initialization.

    angles_between_representative_points : bool, default = False
        Calculate angles between representative points during class initialization.

    Attributes
    ----------
    ds : gpd.GeoDataFrame
        Dataset with block values.

    value_column_name : Any
        Name of the column with block rates.

    geometry_column_name : Any, default = 'geometry'
        Name of the column with a block geometry.

    index_column_name : Any, optional
        Name of the indexing column.

    representative_points_column_name : Any, optional
        The column with representative points or coordinates.

    angles : numpy array
        Angles between the blocks representative points.

    distances : numpy array
        Distances between the blocks representative points.

    block_indexes : numpy array
        Array with blocks indexes.

    block_representative_points : numpy array
        Array with blocks representative points - lon, lat.

    block_values : numpy array
        Array with blocks values.

    Methods
    -------
    block_coordinates(block_id)
        Single block representative point.

    block_real_value(block_id)
        Single block observation.

    calculate_angles_between_representative_points(update=True)
        Angles between blocks, calculated as angles between each representative point and others. If ``update`` is
        True then it updates ``angles`` attribute. Returns dictionary with block id as key and angles to other blocks
        ordered the same way as dictionary keys as values. TODO: change format to DataFrame.

    calculate_distances_between_representative_points(update=True)
        Distances between blocks, calculated as distances between each representative point and others. If ``update``
        is True then it updates ``distances`` attribute. Returns DataFrame with block ids as columns and indexes and
        distances as values.

    get_blocks_values(indexes=None)
        Get multiple blocks values.

    pop(block_index)
        Experimental. Removes block with specified index from the dataset and returns removed block as
        the Blocks object.

    representative_points_array()
        Numpy array with representative points - lon, lat, value.

    transform_crs(target_crs)
        Transform Blocks Coordinate Reference System.




    Examples
    --------

    """

    def __init__(self,
                 ds: gpd.GeoDataFrame,
                 value_column_name,
                 geometry_column_name='geometry',
                 index_column_name=None,
                 representative_points_column_name=None,
                 representative_points_from_centroid=False,
                 representative_points_from_random_sample=False,
                 distances_between_representative_points=True,
                 angles_between_representative_points=False):

        # Helper params
        self._lon_col_name = 'lon'
        self._lat_col_name = 'lat'
        self._default_representative_column_name = 'representative_points'
        self._representative_points_from_centroid = representative_points_from_centroid
        self._representative_points_from_random_sample = representative_points_from_random_sample

        self.ds = ds.copy(deep=True)
        self.value_column_name = value_column_name
        self.index_column_name = index_column_name
        self.geometry_column_name = geometry_column_name
        self.representative_points_column_name = representative_points_column_name

        if representative_points_column_name is None:
            self._get_representative_points()

        # Set lon, lat for further calculations
        self._points_to_floats()

        # get distances & angles
        self.angles = None
        self.distances = None

        if distances_between_representative_points:
            self.distances = self.calculate_distances_between_representative_points(update=False)
        if angles_between_representative_points:
            self.angles = self.calculate_angles_between_representative_points(update=False)

    @property
    def block_indexes(self):
        if self.index_column_name is None:
            return self.ds.index.values
        else:
            return self.ds[self.index_column_name].values

    @property
    def block_representative_points(self):
        return self.ds[[self._lon_col_name, self._lat_col_name]].values

    @property
    def block_values(self):
        return self.ds[self.value_column_name].values

    def block_coordinates(self, block_id: Hashable):
        """
        Gets block representative point.

        Parameters
        ----------
        block_id : Hashable

        Returns
        -------
        : Point
        """
        if self.index_column_name is None:
            ds = self.ds.loc[block_id]
        else:
            ds = self.ds[self.ds[self.index_column_name] == block_id]

        coordinates = ds[self.representative_points_column_name].to_numpy()[0]
        return coordinates

    def block_real_value(self, block_id: Hashable):
        """
        Gets block total value.

        Parameters
        ----------
        block_id : Hashable

        Returns
        -------
        : float
        """
        if self.index_column_name is None:
            ds = self.ds.loc[block_id]
        else:
            ds = self.ds[self.ds[self.index_column_name] == block_id]

        value = float(ds[self.value_column_name].to_numpy()[0])

        return value

    def calculate_angles_between_representative_points(self, update=True) -> Dict:
        """
        Angles from each representative point to others.

        Parameters
        ----------
        update : bool, default = True
            Update ``angles`` attribute.

        Returns
        -------
        : Dict
            block id: angles to other blocks ordered like block ids in a dictionary
        """
        points = self.representative_points_array()
        angles = {}
        indexes = self.block_indexes

        for idx, pt in enumerate(points):
            angle = calc_angles(
                points_b=points,
                origin=pt
            )
            angles[indexes[idx]] = angle

        if update:
            self.angles = angles

        return angles

    def calculate_distances_between_representative_points(self, update=True) -> pd.DataFrame:
        """
        Gets distances between representative points within blocks.

        Parameters
        ----------
        update : bool, default = True
            Update ``distances`` attribute.

        Returns
        -------
        : DataFrame
            Columns and indexes are blocks ids, values are distances between blocks.
        """
        points = self.representative_points_array()
        distances: np.ndarray = point_distance(
            points=points,
            other=points
        )
        indexes = self.block_indexes

        df = pd.DataFrame(data=distances, index=indexes, columns=indexes)

        if update:
            self.distances = df

        return df

    def get_blocks_values(self, indexes: ArrayLike = None):
        if indexes is None:
            ds = self.ds.copy()
        else:
            if self.index_column_name is None:
                ds = self.ds.loc[indexes]
            else:
                ds = self.ds[self.ds[self.index_column_name].isin(indexes)]
        return ds[self.value_column_name].values

    def pop(self, block_index: Union[str, Hashable]):
        """Removes block with specified index from the dataset and returns removed block as the Blocks object"""
        # Get block
        rblock = self._get(block_index)

        # remove block
        self._delete(block_index)

        # return
        return rblock

    def representative_points_array(self):
        return self.ds[[self._lon_col_name, self._lat_col_name, self.value_column_name]].to_numpy(copy=True)

    # TODO manage copying and inplace transformations
    def transform_crs(self, target_crs):
        """Function transforms Blocks CRS

        Parameters
        ----------
        target_crs :
            The target CRS.
        """
        # Transform core dataset
        self.ds.to_crs(target_crs, inplace=True)

        # representative points
        self._get_representative_points()
        self._points_to_floats()

        # distances
        self.distances = self.calculate_distances_between_representative_points(update=False)

    def _delete(self, block_index: Union[str, Hashable]):
        if self.index_column_name is None:
            self.ds.drop(index=block_index, inplace=True)
        else:
            self.ds = self.ds[self.ds[self.index_column_name] != block_index]

    def _get(self, block_index: Union[str, Hashable]):
        if self.index_column_name is None:
            rblock = self.ds.loc[block_index].copy()
            rblock = Blocks(
                ds=rblock,
                value_column_name=self.value_column_name,
                geometry_column_name=self.geometry_column_name,
                representative_points_column_name=self.representative_points_column_name,
                representative_points_from_centroid=self._representative_points_from_centroid,
                representative_points_from_random_sample=self._representative_points_from_random_sample
            )
            return rblock
        else:
            rblock = self.ds[self.ds[self.index_column_name] == block_index].copy()
            rblock = Blocks(
                ds=rblock,
                value_column_name=self.value_column_name,
                geometry_column_name=self.geometry_column_name,
                representative_points_column_name=self.representative_points_column_name,
                representative_points_from_centroid=self._representative_points_from_centroid,
                representative_points_from_random_sample=self._representative_points_from_random_sample
            )
            return rblock

    def _get_random_representative_points(self):
        self.ds[self._default_representative_column_name] = self.ds[
            self.geometry_column_name
        ].representative_point()
        self.representative_points_column_name = self._default_representative_column_name

    def _get_representative_points(self):
        if self._representative_points_from_random_sample:
            if self._representative_points_from_centroid:
                raise AttributeError(
                    'Please set only one parameter from ``representative_points_from_centroid`` or '
                    '``representative_points_from_random_sample`` to True.')
            self._get_random_representative_points()
            # User hasn't provided own set of points
        else:
            self._get_representative_points_from_centroids()

    def _get_representative_points_from_centroids(self):
        """
        Method estimates representative points as coordinates of given polygons.
        """
        self.ds[self._default_representative_column_name] = self.ds[
            self.geometry_column_name
        ].centroid
        self.representative_points_column_name = self._default_representative_column_name

    def _points_to_floats(self):
        """
        Method creates array with [lon, lat, value]

        Returns
        -------
        : numpy array
        """
        lon, lat = points_to_lon_lat(
            self.ds[self.representative_points_column_name]
        )
        self.ds[self._lon_col_name] = lon
        self.ds[self._lat_col_name] = lat

    def _select_distances_between_blocks(self, block_id, other_blocks=None) -> np.ndarray:
        """
        Method selects distances between specified blocks and all other blocks.

        Parameters
        ----------
        block_id :
            Single block ID or list with IDs to retrieve.

        other_blocks : optional
            Other blocks to get distance to those blocks, if not given then all other blocks are returned.

        Returns
        -------
        : numpy array
            Index is block id, columns are other blocks.
        """
        df = self.distances.copy()

        # get specified values
        if other_blocks is None:
            return df.loc[block_id].to_numpy()
        else:
            try:
                return df.loc[block_id, other_blocks].to_numpy()
            except AttributeError:
                return df.loc[block_id, other_blocks]

    def __len__(self):
        return len(self.ds)
