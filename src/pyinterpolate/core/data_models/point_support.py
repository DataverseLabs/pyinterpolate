from typing import Dict, Tuple, Union, List

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.distance.block import calc_block_to_block_distance
from pyinterpolate.distance.point import point_distance
from pyinterpolate.transform.geo import points_to_lon_lat
from pyinterpolate.transform.transform import parse_point_support_distances_array


class PointSupport:
    """
    Class represents blocks and their point support.

    Parameters
    ----------
    points : RawPoints or numpy array
        Point Support as lon, lat, value.

    blocks : Blocks
        Block data.

    store_dropped_points : bool, default = False
        Save dropped points in the additional attribute (``dropped_points``).

    Attributes
    ----------
    point_support : GeoDataFrame
        Columns: ``lon``, ``lat``, ``point-support-value``, ``block-index``.

    dropped_points : numpy array
        ``(lon, lat, value)``

    Notes
    -----
    The PointSupport class structure is designed to store the information about the points within polygons.
    During the regularization process, the inblock variograms are estimated from the polygon's point support, and
    semivariances are calculated between point supports of neighbouring blocks.

    The class takes population grid (support) and blocks data (polygons). Then, spatial join is performed and points
    are assigned to areas within they are placed. The core attribute is ``point_support`` - GeoDataFrame with columns:

    * ``lon`` - a floating representation of longitude,
    * ``lat`` - a floating representation of latitude,
    * ``point-support-value`` - the attribute which describes the name of a column with the point-support's value,
    * ``block-index`` - the name of a column which directs to the block index values.

    """

    def __init__(self,
                 points: gpd.GeoDataFrame,
                 blocks: Blocks,
                 points_value_column: str,
                 points_geometry_column: str,
                 store_dropped_points: bool = False,
                 use_point_support_crs: bool = False,
                 calculate_weighted_block_to_block_distances: bool = True,
                 calculate_distances_between_point_support_points: bool = True,
                 verbose=True):

        self._default_blocks_index_column_name = 'blocks_index'
        self._lon_col_name = 'lon'
        self._lat_col_name = 'lat'
        self._verbose = verbose

        self.dropped_points = None
        self.blocks = blocks
        self.blocks_distances = blocks.distances
        self.blocks_index_column = self.blocks.index_column_name
        self.value_column_name = points_value_column
        self.geometry_column_name = points_geometry_column
        self._total_ps_column_name = f'total_{self.value_column_name}'

        self.point_support_blocks_index_name = None
        self.point_support: gpd.GeoDataFrame
        self.point_support = self._get_point_support(
            points,
            blocks,
            points_value_column,
            points_geometry_column,
            store_dropped_points,
            use_point_support_crs
        )
        self.unique_blocks = self._get_unique_blocks()
        self.weighted_distances = None
        self.distances_between_point_support_points = None

        if calculate_weighted_block_to_block_distances:
            self.weighted_distances = self.weighted_distance(update=False)

        if calculate_distances_between_point_support_points:
            self.calc_distances_between_ps_points(update=True)

    def calc_distances_between_ps_points(self,
                                         update: bool) -> Dict:
        """
        Function calculates distances between all points from point supports within blocks.

        Parameters
        ----------
        update : bool
            Should update class attribute ``distances_between_point_support_points``?

        Returns
        -------
        : Dict
            ``(block_a, block_b): distances``
        """

        data = {}
        for block_a in tqdm(self.unique_blocks, disable=self._verbose):
            points_a = self.get_points_array(block_a)
            for block_b in self.unique_blocks:
                if (block_a, block_b) in data:
                    pass
                else:
                    points_b = self.get_points_array(block_b)
                    # a - rows, b - cols

                    distances: np.ndarray
                    distances = point_distance(points_a[:, :-1],
                                               points_b[:, :-1])

                    out_arr = parse_point_support_distances_array(
                        distances=distances,
                        values_a=points_a[:, -1],
                        values_b=points_b[:, -1]
                    )

                    data[(block_a, block_b)] = out_arr
                    if block_a != block_b:
                        data[(block_b, block_a)] = out_arr

        if update:
            self.distances_between_point_support_points = data
        else:
            return data

    def get_distances_between_known_blocks(self, block_ids: Union[List, np.ndarray]):
        """
        Function returns distances between known blocks.

        Parameters
        ----------
        block_ids : Union[List, np.ndarray]
            List with block indexes.

        Returns
        -------
        : numpy array
            Distances from blocks to all other blocks (ordered the same way as input ``block_ids`` list, where
            rows and columns represent block indexes).
        """
        distances_between_known_blocks = self.blocks.select_distances_between_blocks(
            block_id=block_ids, other_blocks=block_ids
        )
        return distances_between_known_blocks

    def get_point_to_block_indexes(self) -> pd.Series:
        """
        Method returns block indexes for each point in the same order as points are stored in the ``point_support``.

        Returns
        -------
        : pandas Series
            ((point support index: block index))
        """
        return self.point_support[self.blocks_index_column]

    def get_points_array(self, block_id=None) -> np.ndarray:
        """Method returns point coordinates and their values as a numpy array

        Parameters
        ----------
        block_id : Any
            Block for which points shuld be retrieved, if not given then all points are returned.

        Returns
        -------
        : numpy array
            ((lon, lat, value))
        """
        if block_id is None:
            return self.point_support[[self._lon_col_name, self._lat_col_name, self.value_column_name]].to_numpy(
                dtype=np.float32
            )
        else:
            ps = self.point_support[self.point_support[self.point_support_blocks_index_name] == block_id]
            return ps[[self._lon_col_name, self._lat_col_name, self.value_column_name]].to_numpy(
                dtype=np.float32
            )

    def get_weighted_distance(self, block_id):
        if self.weighted_distances is None:
            _ = self.weighted_distance(update=True)

        dists = self.weighted_distances[block_id]
        return dists

    def point_support_totals(self, blocks):
        """
        Function retrieves total point support values for given blocks.

        Parameters
        ----------
        blocks : Iterable
            Block indexes.

        Returns
        -------
        : numpy array
            Retrieved values.
        """
        values = [self._total_point_support_value(bid) for bid in blocks]
        return np.array(values)

    def weighted_distance(self, update=True) -> pd.DataFrame:
        """
        Function calculates weighted distances between blocks using their point supports.

        Parameters
        ----------
        update : bool, default = True
            Update ``weighted_distances`` attribute.

        Returns
        -------
        block_distances : Dict
            Ordered block ids (the order from the list of distances): {block id : [distances to other]}.
        """
        block_distances = calc_block_to_block_distance(
            blocks=self.point_support,
            lon_col_name=self.lon_col_name,
            lat_col_name=self.lat_col_name,
            val_col_name=self.value_column_name,
            block_index_col_name=self.point_support_blocks_index_name
        )

        if update:
            self.weighted_distances = block_distances

        return block_distances

    def _get_point_support(self,
                           points: gpd.GeoDataFrame,
                           blocks: Blocks,
                           points_value_column: str,
                           points_geometry_column: str,
                           store_dropped_points: bool,
                           use_point_support_crs: bool = False) -> gpd.GeoDataFrame:
        """
        Method selects points within blocks. Blocks CRS is used as a baseline.

        Parameters
        ----------
        points : GeoDataFrame

        blocks : Blocks
            Polygon / areas within point support is located.

        points_value_column : str
            The name of a column with points values.

        points_geometry_column : str
            The name of a column with geometry.

        store_dropped_points : bool
            Should object store point support points without linked areas?

        use_point_support_crs : bool, default = False
            Use point support CRS instead of blocks CRS.

        Returns
        -------
        : GeoDataFrame
            ``[lon, lat, point-support-value, point-geometry, block-index]``
        """
        # Transform CRS
        point_support, blocks = self._transform_crs(points, blocks, use_point_support_crs)

        # Merge data
        joined = gpd.sjoin(point_support, blocks.ds, how='left')

        # Check which points weren't joined
        if store_dropped_points:
            is_na = joined.isna().any(axis=1)
            not_joined_points = joined[is_na].copy()
            if len(not_joined_points) > 0:
                self.dropped_points = not_joined_points

        # Clean data
        joined.dropna(inplace=True)
        # TODO: what if name is None?

        joined = self._get_index_geom_val_columns(
            df=joined,
            points_geometry_column=points_geometry_column,
            points_value_column=points_value_column
        )

        # Set lon lat columns
        lon, lat = points_to_lon_lat(
            joined[self.geometry_column_name]
        )
        joined[self._lon_col_name] = lon
        joined[self._lat_col_name] = lat

        # Get total sum of point support values for every block
        totals = joined[[self.point_support_blocks_index_name, self.value_column_name]].groupby(
            self.point_support_blocks_index_name
        ).sum()

        totals.rename(columns={self.value_column_name: self._total_ps_column_name}, inplace=True)
        totals.reset_index(inplace=True)

        joined = pd.merge(joined, totals, how="left", on=self.point_support_blocks_index_name)

        # Set attributes
        return joined

    def _total_point_support_value(self, block_id):
        """
        Function returns total point support value for a given block.

        Parameters
        ----------
        block_id :

        Returns
        -------
        : float
        """
        ds = self.point_support[
            self.point_support[self.point_support_blocks_index_name] == block_id
            ][self._total_ps_column_name].iloc[0]
        return float(ds)

    @staticmethod
    def _transform_crs(points: gpd.GeoDataFrame,
                       blocks: Blocks,
                       use_point_support_crs: bool) -> Tuple[gpd.GeoDataFrame, Blocks]:
        if use_point_support_crs:
            if blocks.ds.crs != points.crs:
                blocks.transform_crs(points.crs)
        else:
            if blocks.ds.crs != points.crs:
                points = points.to_crs(blocks.ds.crs)
        return points, blocks

    def _get_index_geom_val_columns(self,
                                    df: gpd.GeoDataFrame,
                                    points_geometry_column: str,
                                    points_value_column: str):
        """
        Function limits number of columns in a joined dataset.

        Parameters
        ----------
        df : GeoDataFrame
            Joined blocks and point support.

        points_geometry_column : str
            The name of a column with the point support geometry.

        points_value_column : str
            The name of a column with the point support values.

        Returns
        -------
        : GeoDataFrame
            ``[blocks index, points geometry, points values]``
        """

        joined_cols = [points_geometry_column, points_value_column]

        if self.blocks_index_column is None:
            joined_cols.append('index_right')
        else:
            joined_cols.append(self.blocks_index_column)

        df = df[joined_cols].copy()

        if self.blocks_index_column is None:
            df = df.rename(columns={'index_right': self._default_blocks_index_column_name})
            self.point_support_blocks_index_name = self._default_blocks_index_column_name
        else:
            self.point_support_blocks_index_name = self.blocks_index_column

        return df

    def _get_unique_blocks(self) -> np.ndarray:
        """
        Function gets indexes of unique blocks from the point support.

        Returns
        -------
        : numpy array
        """
        unique_blocks = self.point_support[self.point_support_blocks_index_name].unique()
        return unique_blocks

    @property
    def lon_col_name(self):
        return self._lon_col_name

    @property
    def lat_col_name(self):
        return self._lat_col_name
