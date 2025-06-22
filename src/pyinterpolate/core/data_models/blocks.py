"""
Core data structures for block interpolation.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
2. Taher Chegini | @cheginit

Changes in version 1.0
- Package doesn't read data files, data must be loaded into DataFrame and
  then passed into the Blocks object.
"""
from typing import Union, Hashable, Dict
from numpy.typing import ArrayLike

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.distance.angular import calc_angles
from pyinterpolate.distance.point import point_distance
from pyinterpolate.transform.geo import points_to_lon_lat


# TODO: if multipolygon then get coordinates /
#       representative points from the largest block - as an option
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
        Calculate representative points from the point sampled from block
        geometry.

    distances_between_representative_points : bool, default = True
        Calculate distances between representative points during class
        initialization.

    angles_between_representative_points : bool, default = False
        Calculate angles between representative points during class
        initialization.

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

    rep_points_column_name : Any, optional
        The column with representative points or coordinates.

    angles : numpy array
        Angles between the blocks representative points.

    distances : numpy array
        Distances between the blocks representative points.

    Methods
    -------
    block_data()
        Longitude, latitude, and value as numpy array.

    block_indexes()
        Block indexes as numpy array.

    block_representative_points()
        Representative points - lon, lat as numpy array.

    block_values()
        Block values as numpy array.

    block_coordinates(block_id)
        Single block representative point.

    block_real_value(block_id)
        Single block observation.

    calculate_angles_between_rep_points()
        Angles between blocks, calculated as angles between each
        representative point and others. If ``update`` is True then it
        updates ``angles`` attribute. Returns dictionary with block index
        as a key and angles to other blocks ordered the same way as
        dictionary keys as values.

    calculate_distances_between_rep_points()
        Distances between blocks, calculated as distances between each
        representative point and others. If ``update`` is set to True
        then it updates ``distances`` attribute. Returns Data Frame with
        block indexes as columns and indexes and distances as values.

    get_blocks_values()
        Get multiple blocks values.

    pop()
        Experimental. Removes block with specified index from the dataset
        and returns removed block as the ``Blocks`` object. Alters object.

    representative_points_array()
        Numpy array with representative points - longitude, latitude, and
        value.

    select_distances_between_blocks()
        Select distances between a given block and all other blocks.

    transform_crs()
        Transform Blocks Coordinate Reference System.

    Raises
    ------
    AttributeError : If both ``representative_points_from_centroid`` and
                    ``representative_points_from_random_sample`` are set to
                    True.

    See Also
    --------
    PointSupport : Class heavily using ``Blocks`` for
                   the semivariogram deconvolution.

    Examples
    --------
    >>> import os
    >>> import geopandas as gpd
    >>> from pyinterpolate import Blocks
    >>>
    >>>
    >>> FILENAME = 'cancer_data.gpkg'
    >>> LAYER_NAME = 'areas'
    >>> DS = gpd.read_file(FILENAME, layer=LAYER_NAME)
    >>> AREA_VALUES = 'rate'
    >>> AREA_INDEX = 'FIPS'
    >>> AREA_GEOMETRY = 'geometry'
    >>>
    >>> CANCER_DATA = {
    ...    'ds': DS,
    ...    'index_column_name': AREA_INDEX,
    ...    'value_column_name': AREA_VALUES,
    ...    'geometry_column_name': AREA_GEOMETRY
    ... }
    >>> block = Blocks(**CANCER_DATA)
    >>> print(block.ds.columns)
    Index(['FIPS', 'rate', 'geometry', 'rep_points', 'lon', 'lat'],
          dtype='object')
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

        self.__check_rep_points_params(
            representative_points_from_centroid,
            representative_points_from_random_sample
        )

        self._rep_ps_centroid = representative_points_from_centroid
        self._rep_ps_sample = representative_points_from_random_sample

        self.ds = ds.copy(deep=True)
        self.value_column_name = value_column_name
        self.index_column_name = index_column_name
        self.geometry_column_name = geometry_column_name
        self.rep_points_column_name = representative_points_column_name

        if representative_points_column_name is None:
            self._get_representative_points()

        # Set lon, lat for further calculations
        self._points_to_floats()

        # get distances & angles
        self.angles = None
        self.distances = None

        if distances_between_representative_points:
            self.distances = self.calculate_distances_between_rep_points(
                update=False
            )
        if angles_between_representative_points:
            self.angles = self.calculate_angles_between_rep_points(
                update=False
            )

    @property
    def block_data(self):
        """
        Returns block data.

        Returns
        -------
        : numpy array
            Block data [x, y, value].
        """
        return self.ds[
            [self._lon_col_name, self._lat_col_name, self.value_column_name]
        ].to_numpy()

    @property
    def block_indexes(self):
        """
        Returns index column values.

        Returns
        -------
        : numpy array
            Block indexes.
        """
        if self.index_column_name is None:
            return self.ds.index.values
        else:
            return self.ds[self.index_column_name].values

    @property
    def block_representative_points(self):
        """
        Returns block representative coordinates.

        Returns
        -------
        : numpy array
            Block representative coordinates.
        """
        return self.ds[[self._lon_col_name, self._lat_col_name]].values

    @property
    def block_values(self):
        """
        Returns block values.

        Returns
        -------
        : numpy array
            Block values.
        """
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

        coordinates = ds[self.rep_points_column_name].to_numpy()[0]
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

    def calculate_angles_between_rep_points(self, update=True) -> Dict:
        """
        Angles between all representative points to all other
        representative points.

        Parameters
        ----------
        update : bool, default = True
            Update ``angles`` attribute.

        Returns
        -------
        : Dict
            block index: angles to other blocks ordered like block
            indexes (keys) in a dictionary
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

    def calculate_distances_between_rep_points(self,
                                               update=True) -> pd.DataFrame:
        """
        Gets distances between representative points within blocks.

        Parameters
        ----------
        update : bool, default = True
            Update ``distances`` attribute.

        Returns
        -------
        : DataFrame
            Columns and indexes are blocks ids, values are
            distances between blocks.
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
        """
        Returns values of observations aggregated within blocks.

        Parameters
        ----------
        indexes : Array-like, optional
            Indexes of blocks to get values from. If not given then
            all blocks are returned.

        Returns
        -------
        : numpy array
        """
        if indexes is None:
            ds = self.ds.copy()
        else:
            if self.index_column_name is None:
                ds = self.ds.loc[indexes]
            else:
                ds = self.ds[self.ds[self.index_column_name].isin(indexes)]
        return ds[self.value_column_name].values

    def pop(self, block_index: Union[str, Hashable]):
        """
        Removes block with specified index from the dataset and returns
        removed block as the ``Blocks`` object.

        Parameters
        ----------
        block_index : Union[str, Hashable]
            Index of the block to remove.

        Returns
        -------
        : Blocks
            Single block as the Blocks object.
        """
        # Get block
        rblock = self._get(block_index)

        # remove block
        self._delete(block_index)

        # return
        return rblock

    def representative_points_array(self):
        """
        Returns array with blocks' representative points.

        Returns
        -------
        : numpy array
            ``[lon, lat, value]``
        """

        result = self.ds[
            [self._lon_col_name, self._lat_col_name, self.value_column_name]
        ].to_numpy(copy=True)

        return result

    def select_distances_between_blocks(self,
                                        block_id,
                                        other_blocks=None) -> np.ndarray:
        """
        Method selects distances between specified blocks and all other
        blocks.

        Parameters
        ----------
        block_id :
            Single block ID or list with IDs to retrieve.

        other_blocks : optional
            Other blocks to get distance to those blocks, if not given then
            all other blocks are returned.

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

    # TODO manage copying and inplace transformations
    def transform_crs(self, target_crs):
        """Function transforms Blocks CRS

        Parameters
        ----------
        target_crs : pyproj.CRS or EPSG code
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input()
            <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        """
        # Transform core dataset
        self.ds.to_crs(target_crs, inplace=True)

        # representative points
        self._get_representative_points()
        self._points_to_floats()

        # distances
        self.distances = self.calculate_distances_between_rep_points(
            update=False
        )

        # angles
        self.angles = self.calculate_angles_between_rep_points(
            update=False
        )

    def _delete(self, block_index: Union[str, Hashable]):
        """
        Removes block with specified index from the dataset.

        Parameters
        ----------
        block_index : Union[str, Hashable]
            Index of the block to remove.
        """
        if self.index_column_name is None:
            self.ds.drop(index=block_index, inplace=True)
        else:
            self.ds = self.ds[self.ds[self.index_column_name] != block_index]

    def _get(self, block_index: Union[str, Hashable]):
        """
        Returns block with specified index from the dataset.

        Parameters
        ----------
        block_index : Union[str, Hashable]
            Index of the block to return.

        Returns
        -------
        : Blocks
            Single block as the Blocks object.
        """
        if self.index_column_name is None:
            rblock = self.ds.loc[block_index].copy()
            rblock = Blocks(
                ds=rblock,
                value_column_name=self.value_column_name,
                geometry_column_name=self.geometry_column_name,
                representative_points_column_name=self.rep_points_column_name,
                representative_points_from_centroid=self._rep_ps_centroid,
                representative_points_from_random_sample=self._rep_ps_sample
            )
            return rblock
        else:
            rblock = self.ds[
                self.ds[self.index_column_name] == block_index
                ].copy()

            rblock = Blocks(
                ds=rblock,
                value_column_name=self.value_column_name,
                geometry_column_name=self.geometry_column_name,
                representative_points_column_name=self.rep_points_column_name,
                representative_points_from_centroid=self._rep_ps_centroid,
                representative_points_from_random_sample=self._rep_ps_sample
            )
            return rblock

    def _get_random_representative_points(self):
        """
        Method estimates representative points as a point guaranteed to be
        within the object, cheaply.
        """
        self.ds[self._default_representative_column_name] = self.ds[
            self.geometry_column_name
        ].representative_point()

        self.rep_points_column_name = self._default_representative_column_name

    def _get_representative_points(self):
        """
        Estimates representative points as centroids or randomly
        sampled points within the block geometry.
        """
        if self._rep_ps_sample:
            self._get_random_representative_points()
        else:
            self._get_representative_points_from_centroids()

    def _get_representative_points_from_centroids(self):
        """
        Estimates representative points as centroids.
        """
        self.ds[self._default_representative_column_name] = self.ds[
            self.geometry_column_name
        ].centroid
        self.rep_points_column_name = self._default_representative_column_name

    def _points_to_floats(self):
        """
        Method creates array with [lon, lat, value]

        Returns
        -------
        : numpy array
        """
        lon, lat = points_to_lon_lat(
            self.ds[self.rep_points_column_name]
        )
        self.ds[self._lon_col_name] = lon
        self.ds[self._lat_col_name] = lat

    def __len__(self):
        """
        Returns number of blocks in the dataset.

        Returns
        -------
        : int
        """
        return len(self.ds)

    @staticmethod
    def __check_rep_points_params(r1, r2):
        """
        Checks if users has chosen only one method to calculate representative
        points.

        Parameters
        ----------
        r1 : bool
            First parameter (``representative_points_from_centroid``).

        r2 : bool
            Second parameter (``representative_points_from_random_sample``).

        Raises
        ------
        AttributeError
            If both parameters are set to ``True``.

        """
        if r1 and r2:
            raise AttributeError(
                'Please set only one parameter from '
                '``representative_points_from_centroid`` or '
                '``representative_points_from_random_sample`` to True.'
            )
