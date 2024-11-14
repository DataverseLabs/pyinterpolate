from typing import Hashable, Union

import numpy as np

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.transform_poisson_kriging_data import block_base_distances, block_to_blocks_angles, \
    set_blocks_dataset


class CentroidPoissonKrigingInput:
    """
    Represents Centroid-based Poisson Kriging input.

    [[unknown block index, cx, cy, value, distance to unknown centroid, angle to the origin]]
    """

    def __init__(self,
                 block_id: Union[str, Hashable],
                 point_support: PointSupport,
                 semivariogram_model: TheoreticalVariogram,
                 blocks_indexes: np.ndarray = None,
                 weighted: bool = True):

        # Neighbors df columns
        self.n_blocks_indexes_column = 'blocks_indexes'
        self.n_points_column = 'points'
        self.n_values_column = 'values'
        self.n_distances_column = 'distances'
        self.n_angular_differences_column = 'angular_differences'
        self.n_block_id_column = 'block_id'
        self.block_total_point_support_column = 'total'

        self.semivariogram_model = semivariogram_model

        self.block_id = block_id
        self.is_weighted = weighted
        self.is_directional = True if self.semivariogram_model.direction is not None else False
        self.angular_tolerance = 1
        self.max_tick = 15

        self.base_distances = block_base_distances(
            block_id=self.block_id,
            point_support=point_support,
            weighted=self.is_weighted
        )

        self.angle_differences = None
        if self.is_directional:
            self.angle_differences = block_to_blocks_angles(
                block_id=self.block_id,
                point_support=point_support,
                direction=semivariogram_model.direction
            )

        self.ds = set_blocks_dataset(
            block_id=self.block_id,
            points=point_support.blocks.block_representative_points,
            values=point_support.blocks.block_values,
            distances=self.base_distances,
            blocks_indexes=blocks_indexes,
            angular_differences=self.angle_differences,
            angular_differences_column=self.n_angular_differences_column,
            blocks_indexes_column=self.n_blocks_indexes_column,
            core_block_id_column=self.n_block_id_column,
            distances_column=self.n_distances_column,
            points_column=self.n_points_column,
            values_column=self.n_values_column
        )

        # Neighbors finding parameters
        self._max_range = None
        self._min_number_of_neighbors = None
        self._select_all_possible_neighbors = None

        # kriging input
        self._kriging_input = None

    @property
    def coordinates(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, please select neighbors first.')
        else:
            return self._kriging_input[self.n_points_column].to_numpy()

    @property
    def distances(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, please select neighbors first.')
        else:
            return self._kriging_input[self.n_distances_column].to_numpy()

    @property
    def kriging_input(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, please select neighbors first.')
        else:
            return self._kriging_input.copy()

    @property
    def neighbors_indexes(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, please select neighbors first.')
        else:
            return self._kriging_input.index.values

    @property
    def pk_input(self):
        return self.ds.copy()

    @property
    def values(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, please select neighbors first.')
        else:
            return self._kriging_input[self.n_values_column].to_numpy()

    def select_neighbors(self,
                         max_range: float,
                         min_number_of_neighbors: int,
                         select_all_possible_neighbors: bool = True):
        """
        Function selects the closest neighbors.

        Parameters
        ----------
        max_range : float
            The maximum distance for the neighbors search.

        min_number_of_neighbors : int
            The minimum number of neighboring areas.

        select_all_possible_neighbors : bool, default = True
            Should select all possible neighbors or only the ``min_number_of_neighbors``?

        Returns
        -------
        : DataFrame
        """
        self._update_private_neighbors_search_params(
            max_range=max_range,
            min_number_of_neighbors=min_number_of_neighbors,
            select_all_possible_neighbors=select_all_possible_neighbors
        )
        ds = self._select_neighbors_in_range(max_range=max_range)

        if len(ds) == 0:
            raise ValueError(f'For a given distance: {max_range} no neighbors have been found.')

        self._kriging_input = self._select_required_number_of_neighbors(
            df=ds,
            min_number_of_neighbors=min_number_of_neighbors,
            select_all_possible_neighbors=select_all_possible_neighbors
        )

    def select_neighbors_directional(self,
                                     max_range: float,
                                     min_number_of_neighbors: int,
                                     select_all_possible_neighbors: bool = True,
                                     angular_tolerance: float = None,
                                     max_tick: float = None):
        """
        Function selects neighbors to a point using angle (direction) between neighbors as additional filtering
        parameter.

        Parameters
        ----------
        max_range : float
            The maximum distance for the neighbors search.

        min_number_of_neighbors : int
            The minimum number of neighboring areas.

        select_all_possible_neighbors : bool, default = True
            Should select all possible neighbors or only the ``min_number_of_neighbors``?

        angular_tolerance : float, optional
            How many degrees of difference are allowed for the neighbors search.

        max_tick : float, optional
            How wide might the ``angular_tolerance`` (in one direction, in reality it is two times greater, up to
            180 -> 360 degrees)

        Returns
        -------
        : DataFrame
        """

        self._update_private_neighbors_search_params(
            max_range=max_range,
            min_number_of_neighbors=min_number_of_neighbors,
            select_all_possible_neighbors=select_all_possible_neighbors
        )

        if self.n_angular_differences_column in self.ds.columns:
            self._update_neighbors_search_params_directional(
                angular_tolerance=angular_tolerance,
                max_tick=max_tick
            )

            # First select only neighbors in a max range
            ds = self._select_neighbors_in_range(max_range=max_range)

            # Next, get points in a specific angle
            ads = ds[ds[self.n_angular_differences_column] <= self.angular_tolerance]

            while len(ads) < min_number_of_neighbors:
                self.angular_tolerance = self.angular_tolerance + 1
                ads = ds[ds[self.n_angular_differences_column] <= self.angular_tolerance]
                if self.max_tick < self.angular_tolerance:
                    break

            if len(ads) == 0:
                raise ValueError('For a given distances and angle no neighbors have been found. Make ``max_tick`` '
                                 'larger or search for neighbors in the every direction.')

            self._kriging_input = self._select_required_number_of_neighbors(
                df=ads,
                min_number_of_neighbors=min_number_of_neighbors,
                select_all_possible_neighbors=select_all_possible_neighbors
            )

        else:
            raise AttributeError('Angular differences are not provided, cannot find neighbors. Please, initialize '
                                 'class instance with the ``angular_differences`` array.')

    def _select_neighbors_in_range(self, max_range):
        ds = self.ds.copy(deep=True)

        ds = ds[ds[self.n_distances_column] <= max_range]
        ds = ds[ds[self.n_distances_column] > 0]
        ds.sort_values(by=self.n_distances_column, ascending=True, inplace=True)
        return ds

    def _select_required_number_of_neighbors(self, df, min_number_of_neighbors, select_all_possible_neighbors):
        if select_all_possible_neighbors:
            return df
        else:
            if len(df) >= min_number_of_neighbors:
                return df.sort_values(by=self.n_distances_column, ascending=True).iloc[:min_number_of_neighbors]
            else:
                return df.sort_values(by=self.n_distances_column, ascending=True)

    def _update_neighbors_search_params_directional(self,
                                                    angular_tolerance,
                                                    max_tick):
        if angular_tolerance is not None:
            self.angular_tolerance = angular_tolerance

        if max_tick is not None:
            self.max_tick = max_tick

    def _update_private_neighbors_search_params(self,
                                                max_range,
                                                min_number_of_neighbors,
                                                select_all_possible_neighbors):
        self._max_range = max_range
        self._select_all_possible_neighbors = select_all_possible_neighbors
        self._min_number_of_neighbors = min_number_of_neighbors

    def __len__(self):
        if self._kriging_input is None:
            return 0
        else:
            return len(self._kriging_input)
