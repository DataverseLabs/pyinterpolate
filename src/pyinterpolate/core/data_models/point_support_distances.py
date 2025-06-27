from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyinterpolate.distance.block import calc_block_to_block_distance
from pyinterpolate.distance.point import point_distance
from pyinterpolate.transform.transform import \
    parse_point_support_distances_array


class PointSupportDistance:
    """
    Class calculates and stores distances between point supports of multiple
    blocks.

    Parameters
    ----------
    verbose : bool, default = True
        Show progress.

    Attributes
    ----------
    weighted_block_to_block_distances : DataFrame
        Indexes: block indexes, Columns: block indexes, Cells: distances.

    distances_between_point_supports : Dict
        ``(block_a, block_b): [[value_a(i), value_b(j), distance(i-j)], ...]``

    no_closest_neighbors : int
        Number of closest neighbors for each block.

    closest_neighbors : Dict
        Block id: [the closest blocks].


    Methods
    -------
    calc_pair_distances()
        Returns distances between point supports from two blocks and
        updates distances dictionary.

    calculate_point_support_distances()
        Calculates distances between point supports.

    calculate_weighted_block_to_block_distances()
        Calculates weighted distances between blocks using their
        point supports.

    get_weighted_distance()
        Returns weighted distance to a block.

    Raises
    ------
    AttributeError :
        When weighted block to block distances are not calculated and user
        wants to find closest neighbors (using
        ``calculate_point_support_distances()`` method with
        ``no_closest_neighbors`` > 0).
    """

    def __init__(self, verbose=True):

        self.weighted_block_to_block_distances: pd.DataFrame = None
        self.distances_between_point_supports = dict()

        self.no_closest_neighbors = 0
        self.closest_neighbors = dict()

        self.verbose = verbose

        self._block_indexes: List = None
        self._calculated_block_pairs = set()

    def calc_pair_distances(self,
                            point_support,
                            block_pair: Tuple,
                            update=True):
        """
        Returns distances between point supports from two blocks and
        updates distances dictionary.

        Parameters
        ----------
        point_support : PointSupport
            Blocks and their point supports.

        block_pair : Tuple
            (block_a, block_b)

        update : bool, default = True
            If True then distances are updated in the distances' dictionary.

        Returns
        -------
        : numpy array
            Distances between point supports from two blocks.
        """

        if block_pair in self._calculated_block_pairs:
            return self.distances_between_point_supports[block_pair]

        block_a = block_pair[0]
        block_b = block_pair[1]

        points_a = point_support.get_points_array(block_a)
        points_b = point_support.get_points_array(block_b)

        # a - rows, b - cols
        out_arr = self._point_support_distances_between_blocks(points_a,
                                                               points_b)

        # Update distances
        if update:
            if block_pair not in self._calculated_block_pairs:
                self.distances_between_point_supports.update(
                    {block_pair: out_arr}
                )
                self._calculated_block_pairs.add(block_pair)
                self.distances_between_point_supports.update(
                    {(block_b, block_a): out_arr}
                )
                self._calculated_block_pairs.add((block_b, block_a))

        return out_arr

    def calculate_point_support_distances(self,
                                          point_support,
                                          block_id,
                                          no_closest_neighbors: int = 0):
        """
        Calculates distances between point supports.

        Parameters
        ----------
        point_support : PointSupport
            Blocks and their point supports.

        block_id : int
            The unique id of a block.

        no_closest_neighbors : int, default = 0
            Number of the closest neighbors. If default then all distances
            are returned.

        Returns
        -------
        : Dict
            Dictionary with distances between point supports of a given
            block and its neighbors.
            Key is a block pair, and value is a numpy array with distances,
            where each row represents a point from a given block and each
            column represents a point from its neighbor.
        """
        if self.verbose:
            print('Calculating distances between point supports...')

        if no_closest_neighbors > 0:
            self.no_closest_neighbors = no_closest_neighbors

        _d_p_bar = not self.verbose

        data = self._calc_distances_between_ps_points_neighbors(point_support,
                                                                block_id,
                                                                _d_p_bar)

        return data

    def calculate_weighted_block_to_block_distances(self,
                                                    point_support,
                                                    return_distances=False):
        """
        Calculates weighted distances between blocks using their
        point supports.

        Parameters
        ----------
        point_support : PointSupport
            Blocks and their point supports.

        return_distances : bool, default = False
            Should return DataFrame with distances?

        Returns
        -------
        : pd.DataFrame
            Indexes: block indexes, Columns: block indexes, Cells: distances.
        """
        if self.verbose:
            print('Calculating weighted distances between blocks...')

        block_distances = calc_block_to_block_distance(
            ps_blocks=point_support,
            lon_col_name=point_support.lon_col_name,
            lat_col_name=point_support.lat_col_name,
            val_col_name=point_support.value_column_name,
            block_id_col_name=point_support.point_support_blocks_index_name
        )

        # block_distances : pandas DataFrame
        #   Indexes: block indexes, Columns: block indexes, Cells: distances.
        self.weighted_block_to_block_distances = block_distances

        if return_distances:
            return block_distances

    def get_weighted_distance(self, block_id) -> pd.Series:
        """
        Returns weighted distance to a block.

        Parameters
        ----------
        block_id : Union[Hashable, str]
            Block unique index.

        Returns
        -------
        : pd.Series
            Weighted distances between the block and other blocks centroids.
        """

        dists = self.weighted_block_to_block_distances[block_id]
        return dists

    def _calc_distances_between_ps_points_all(self,
                                              point_support,
                                              disable_progress_bar: bool):
        """
        Calculates distances between point supports of all blocks.

        Parameters
        ----------
        point_support : PointSupport
            Blocks and their point supports.

        disable_progress_bar : bool
            Disable progress bar.

        Returns
        -------
        : Dict
            Dictionary with distances between point supports of all blocks.
            Key is a block pair, and value is a numpy array with distances
            and point support values,
            where each row represents a point from the first block and each
            column represents a point from the second block.
        """
        data = {}
        for block_a in tqdm(point_support.unique_blocks,
                            disable=disable_progress_bar):
            points_a = point_support.get_points_array(block_a)
            for block_b in point_support.unique_blocks:
                if (block_a, block_b) in data:
                    pass
                else:
                    points_b = point_support.get_points_array(block_b)

                    # a - rows, b - cols
                    out_arr = self._point_support_distances_between_blocks(
                        points_a, points_b
                    )

                    data[(block_a, block_b)] = out_arr
                    if block_a != block_b:
                        data[(block_b, block_a)] = out_arr
        return data

    def _calc_distances_between_ps_points_neighbors(self,
                                                    point_support,
                                                    block_id,
                                                    disable_progress_bar):
        """
        Function calculates distances between point supports of neighbouring
        blocks.

        Parameters
        ----------
        point_support : PointSupport
            Blocks and their point supports.

        block_id : Union[Hashable, str]
            Unique index of a block for which distances to its neighboring
            blocks are calculated.

        disable_progress_bar : bool
            Disable progress bar.

        Returns
        -------
        : Dict
            Dictionary with distances between point supports of a given
            block and its neighbors.
            Key is a block pair, and value is a numpy array with distances
            and point support values,
            where each row represents a point from a given block and each
            column represents a point from its neighbor.
        """
        possible_neighbors = self._find_closest_neighbors(block_id)
        base_areas = list(possible_neighbors.keys())
        data = {}

        for block_a in tqdm(base_areas, disable=disable_progress_bar):
            points_a = point_support.get_points_array(block_a)
            for block_b in possible_neighbors[block_a]:
                if (block_a, block_b) not in data:
                    points_b = point_support.get_points_array(block_b)

                    # a - rows, b - cols
                    out_arr = self._point_support_distances_between_blocks(
                        points_a, points_b
                    )

                    data[(block_a, block_b)] = out_arr
                    if block_a != block_b:
                        data[(block_b, block_a)] = out_arr

        return data

    def _find_closest_neighbors(self, block_id):
        """
        Function calculates possible neighbors of a given block.

        Parameters
        ----------
        block_id : Hashable
            Block id.

        Returns
        -------
        : Dict
            {block_id: [possible neighbors]}
        """
        if self.weighted_block_to_block_distances is None:
            raise AttributeError(
                'Calculate weighted block to block distances first using '
                '".calculate_weighted_block_to_block_distances()" method.'
            )

        # series: block id - distance
        df = self.weighted_block_to_block_distances.loc[block_id]
        df = df.sort_values()
        if self.no_closest_neighbors > 0:
            dlist = df.index.values[1:self.no_closest_neighbors + 1].tolist()
        else:
            dlist = df.index.values[1:].tolist()
        df_dict = {block_id: dlist}

        self.closest_neighbors.update(df_dict)

        return df_dict

    @staticmethod
    def _point_support_distances_between_blocks(points_a, points_b):
        """
        Function calculates distances between point supports of two blocks.

        Parameters
        ----------
        points_a : numpy array
            Coordinates of point support points from the first block.

        points_b : numpy array
            Coordinates of point support points from the second block.

        Returns
        -------
        : numpy array
            ``[[value_a(i), value_b(j), distance(i-j)]]``
        """
        distances: np.ndarray
        distances = point_distance(points_a[:, :-1],
                                   points_b[:, :-1])

        out_arr = parse_point_support_distances_array(
            distances=distances,
            values_a=points_a[:, -1],
            values_b=points_b[:, -1]
        )
        return out_arr
