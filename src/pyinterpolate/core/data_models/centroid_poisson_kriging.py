from typing import Hashable, Union

import numpy as np

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.point_support_distances import PointSupportDistance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.transform_poisson_kriging_data import block_base_distances, block_to_blocks_angles, \
    set_blocks_dataset


class CentroidPoissonKrigingInput:
    """
    Represents Centroid-based Poisson Kriging input.

    Parameters
    ----------
    block_id : Union[str, Hashable]
        Index of the interpolated block.

    point_support : PointSupport
        Point support object - containing interpolated block ``block_id``
        and neighboring ps_blocks.

    semivariogram_model : TheoreticalVariogram
        Fitted semivariogram.

    blocks_indexes : np.ndarray, optional
        Known ps_blocks (all possible neighbors) indexes. Parameter is optional,
        because the class can index neighbors based on their representative
        coordinates.

    weighted : bool, default = True
        Should distances be weighted by the point support values?

    verbose : bool, default = False
        Should the class print additional information?

    Attributes
    ----------
    semivariogram_model : TheoreticalVariogram
        Fitted semivariogram.

    block_id : Union[str, Hashable]
        ID of the interpolated block.

    is_weighted : bool
        Should distances be weighted by the point support values?

    is_directional : bool
        Is the semivariogram directional? Set to ``True`` if passed
        ``semivariogram_model`` has a direction.

    angular_tolerance : float, default = 1
        How many degrees of difference are allowed for the neighbors search.

    max_tick : float, default = 15
        How wide might the ``angular_tolerance`` (in one direction,
        in reality it is two times greater).

    base_distances : numpy array
        The distances to the unknown block ``block_id`` from other ps_blocks.

    angle_differences : numpy array, default = None
        The angles between the unknown block and other ps_blocks. It is
        calculated only for a directional semivariogram.

    ds : DataFrame
        Core structure within the class. It contains all the necessary data
        for the kriging. DataFrame columns are:
        ``['points', 'values', 'distances', 'angular_differences', 'block_id']``

    Methods
    -------
    angles : numpy array, property
        Get angles between the neighboring ps_blocks and the unknown block.

    coordinates : numpy array, property
        Get coordinates of the neighboring ps_blocks.

    distances : numpy array, property
        Get distances to the neighboring ps_blocks from the unknown block.

    kriging_input : DataFrame, property
        DataFrame representing unknown block's neighbors.

    neighbors_indexes : numpy array, property
        Get indexes of the neighboring ps_blocks.

    pk_input : DataFrame, property
        DataFrame representing all ps_blocks and their relations to the unknown
        block. (See ``ds`` attribute).

    values : numpy array, property
        Get values of the neighboring ps_blocks.

    select_neighbors()
        Function selects the closest neighbors.

    select_neighbors_directional()
        Function selects the closest neighbors within a defined angle.

    len()
        Returns the number of closest neighbors.

    Raises
    ------
    AttributeError :
        - Angular differences were not provided and user wants to select
          neighbors based on direction.

    ValueError :
        - Kriging input is not calculated and user tries to access ``angles``,
          ``coordinates``, ``distances``, ``kriging_input`` and
          ``neighbors_indexes`` properties.
        - For a given distance or angle no neighbors have been found.


    See Also
    --------
    select_centroid_poisson_kriging_data() :
        function which uses this class to select neighbors for the
        interpolation.

    Examples
    --------
    >>> import os
    >>> import geopandas as gpd
    >>> from pyinterpolate import (
    >>> Blocks, ExperimentalVariogram, PointSupport, TheoreticalVariogram
    >>> )
    >>> from pyinterpolate.core.data_models.centroid_poisson_kriging import (
    >>> CentroidPoissonKrigingInput
    >>> )
    >>>
    >>>
    >>> FILENAME = 'cancer_data.gpkg'
    >>> LAYER_NAME = 'areas'
    >>> DS = gpd.read_file(FILENAME, layer=LAYER_NAME)
    >>> AREA_VALUES = 'rate'
    >>> AREA_INDEX = 'FIPS'
    >>> AREA_GEOMETRY = 'geometry'
    >>> PS_LAYER_NAME = 'points'
    >>> PS_VALUES = 'POP10'
    >>> PS_GEOMETRY = 'geometry'
    >>> PS = gpd.read_file(FILENAME, layer=PS_LAYER_NAME)
    >>>
    >>> CANCER_DATA = {
    ...    'ds': DS,
    ...    'index_column_name': AREA_INDEX,
    ...    'value_column_name': AREA_VALUES,
    ...    'geometry_column_name': AREA_GEOMETRY
    ... }
    >>> POINT_SUPPORT_DATA = {
    ...     'ps': PS,
    ...     'value_column_name': PS_VALUES,
    ...     'geometry_column_name': PS_GEOMETRY
    ... }
    >>> BLOCKS = Blocks(**CANCER_DATA)
    >>> indexes = BLOCKS.block_indexes
    >>>
    >>> PS = PointSupport(
    ...     points=POINT_SUPPORT_DATA['ps'],
    ...     ps_blocks=BLOCKS,
    ...     points_value_column=POINT_SUPPORT_DATA['value_column_name'],
    ...     points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    ... )
    >>>
    >>> EXPERIMENTAL = ExperimentalVariogram(
    ...     ds=BLOCKS.representative_points_array(),
    ...     step_size=40000,
    ...     max_range=300001
    ... )
    >>>
    >>> THEO = TheoreticalVariogram()
    >>> THEO.autofit(
    ...     experimental_variogram=EXPERIMENTAL,
    ...     sill=150
    ... )
    >>> cpki = CentroidPoissonKrigingInput(
    ...     block_id=indexes[-5],
    ...     point_support=PS,
    ...     semivariogram_model=THEO
    ... )
    >>> cpki.select_neighbors(
    ...     max_range=120000,
    ...     min_number_of_neighbors=4,
    ...     select_all_possible_neighbors=False
    ... )
    >>> print(len(cpki))  # 3 neighbors are selected from this dataset
    3
    """

    def __init__(self,
                 block_id: Union[str, Hashable],
                 point_support: PointSupport,
                 semivariogram_model: TheoreticalVariogram,
                 blocks_indexes: np.ndarray = None,
                 weighted: bool = True,
                 verbose: bool = False):

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
        self.is_directional = (
            True if self.semivariogram_model.direction is not None else False
        )
        self.angular_tolerance = None
        self.max_tick = None
        self.ps_dists = PointSupportDistance(
            verbose=verbose
        )

        self.base_distances = block_base_distances(
            block_id=self.block_id,
            point_support=point_support,
            point_support_distances=self.ps_dists
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

        # kriging input
        self._kriging_input = None

    @property
    def angles(self):
        if self._kriging_input is None:
            msg = ('Kriging input is not estimated yet, please '
                   'select neighbors first.')
            raise ValueError(msg)
        else:
            return self._kriging_input[
                self.n_angular_differences_column
            ].to_numpy()

    @property
    def coordinates(self):
        if self._kriging_input is None:
            msg = ('Kriging input is not estimated yet, '
                   'please select neighbors first.')
            raise ValueError(msg)
        else:
            return self._kriging_input[self.n_points_column].to_numpy()

    @property
    def distances(self):
        if self._kriging_input is None:
            msg = ('Kriging input is not estimated yet, '
                   'please select neighbors first.')
            raise ValueError(msg)
        else:
            return self._kriging_input[self.n_distances_column].to_numpy()

    @property
    def kriging_input(self):
        if self._kriging_input is None:
            msg = ('Kriging input is not estimated yet, '
                   'please select neighbors first.')
            raise ValueError(msg)
        else:
            return self._kriging_input.copy()

    @property
    def neighbors_indexes(self):
        if self._kriging_input is None:
            msg = ('Kriging input is not estimated yet, '
                   'please select neighbors first.')
            raise ValueError(msg)
        else:
            return self._kriging_input.index.values

    @property
    def pk_input(self):
        return self.ds.copy()

    @property
    def values(self):
        if self._kriging_input is None:
            msg = ('Kriging input is not estimated yet, '
                   'please select neighbors first.')
            raise ValueError(msg)
        else:
            return self._kriging_input[self.n_values_column].to_numpy()

    def select_neighbors(self,
                         max_range: float,
                         min_number_of_neighbors: int):
        """
        Function selects the closest neighbors.

        Parameters
        ----------
        max_range : float
            The maximum distance for the neighbors search.

        min_number_of_neighbors : int
            The minimum number of neighboring areas.
        """
        self._update_private_neighbors_search_params(
            max_range=max_range,
            min_number_of_neighbors=min_number_of_neighbors
        )
        self._kriging_input = self._select_min_no_neighbors(
            no_neighbors=min_number_of_neighbors
        )

    def select_neighbors_directional(self,
                                     max_range: float,
                                     min_number_of_neighbors: int,
                                     angular_tolerance: float = 1,
                                     max_tick: float = 15):
        """
        Function selects neighbors to a point using angle
        (direction) between neighbors as additional filtering parameter.

        Parameters
        ----------
        max_range : float
            The maximum distance for the neighbors search.

        min_number_of_neighbors : int
            The minimum number of neighboring areas.

        select_all_possible_neighbors : bool, default = True
            Should select all possible neighbors or only the
            ``min_number_of_neighbors``?

        angular_tolerance : float, optional
            How many degrees of difference are allowed for the neighbors
            search.

        max_tick : float, optional
            How wide might the ``angular_tolerance`` (in one direction,
            in reality it is two times greater, up to 180 -> 360 degrees)
        """
        self._update_private_neighbors_search_params(
            max_range=max_range,
            min_number_of_neighbors=min_number_of_neighbors
        )

        if self.n_angular_differences_column in self.ds.columns:
            self._update_neighbors_search_params_directional(
                angular_tolerance=angular_tolerance,
                max_tick=max_tick
            )

            # First select only neighbors in a max range
            ds = self._sort_neighbors()
            ds = ds[ds[self.n_distances_column] <= self._max_range]

            # Next, get points in a specific angle
            ads = self._points_within_angle(ds)

            while len(ads) < min_number_of_neighbors:
                self.angular_tolerance = self.angular_tolerance + 1
                ads = self._points_within_angle(ds)

                if self.max_tick < self.angular_tolerance:
                    break

            if len(ads) == 0:
                raise ValueError('For a given distances and angle '
                                 'no neighbors have been found. '
                                 'Make ``max_tick`` '
                                 'larger or search for neighbors in '
                                 'the every direction.')

            self._kriging_input = ads

        else:
            raise AttributeError('Angular differences are not provided, '
                                 'cannot find neighbors. Please, initialize '
                                 'class instance with '
                                 'the ``angular_differences`` array.')

    def _points_within_angle(self, ds):
        """
        Function selects points within a given angle.

        Parameters
        ----------
        ds : DataFrame
            Potential neighbors and angles between them.

        Returns
        -------

        """
        ads = ds[
            ds[self.n_angular_differences_column] <= self.angular_tolerance]

        return ads

    def _sort_neighbors(self):
        """
        Selects neighbors within a given range.

        Returns
        -------
        : DataFrame
            Potential neighbors and distances to them.
        """

        ds = self.ds[self.ds[self.n_distances_column] > 0].copy(deep=True)
        ds.sort_values(
            by=self.n_distances_column, ascending=True, inplace=True
        )
        return ds

    def _select_min_no_neighbors(self, no_neighbors):
        """
        Selects neighbors within a given range.

        Parameters
        ----------
        no_neighbors : int
            Number of Kriging system neighbors.

        Returns
        -------
        : DataFrame
            Potential neighbors and distances to them.
        """

        ds = self._sort_neighbors()
        return ds.iloc[:no_neighbors]

    def _update_neighbors_search_params_directional(self,
                                                    angular_tolerance,
                                                    max_tick):
        """
        Updates private parameters related to angles between neighbors
        for the neighbors search.

        Parameters
        ----------
        angular_tolerance : float
            How many degrees of difference are allowed for the
            neighbors search.

        max_tick : float
            How wide might the ``angular_tolerance`` (in one direction,
            in reality it is two times greater).
        """
        if angular_tolerance is not None:
            self.angular_tolerance = angular_tolerance

        if max_tick is not None:
            self.max_tick = max_tick

    def _update_private_neighbors_search_params(self,
                                                max_range,
                                                min_number_of_neighbors):
        """
        Updates private parameters for the neighbors search.

        Parameters
        ----------
        max_range : float
            The maximum distance for the neighbors search.

        min_number_of_neighbors : int
            The minimum number of neighboring areas.
        """
        self._max_range = max_range
        self._min_number_of_neighbors = min_number_of_neighbors

    def __len__(self):
        """
        Returns the number of closest neighbors.

        Returns
        -------
        : int
        """
        if self._kriging_input is None:
            return 0
        else:
            return len(self._kriging_input)
