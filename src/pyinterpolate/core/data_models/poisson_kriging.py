from typing import Hashable, Union

import numpy as np
import pandas as pd

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.point_support_distances import PointSupportDistance
from pyinterpolate.distance.angular import calc_angles_between_points
from pyinterpolate.distance.point import point_distance
from pyinterpolate.semivariogram.deconvolution.block_to_block_semivariance import weighted_avg_point_support_semivariances
from pyinterpolate.semivariogram.deconvolution.point_to_block_semivariance import calculate_average_p2b_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.transform_poisson_kriging_data import block_base_distances, block_to_blocks_angles, \
    set_blocks_dataset, parse_kriging_input


class PoissonKrigingInput:
    r"""
    Represents Poisson Kriging input.

    Parameters
    ----------
    block_id : Union[str, Hashable]
        ID of the interpolated block.

    point_support : PointSupport
        Point support object - containing interpolated block ``block_id``
        and neighboring ps_blocks.

    semivariogram_model : TheoreticalVariogram
        Fitted semivariogram.

    blocks_indexes : np.ndarray, optional
        The list of the known ps_blocks names.

    Attributes
    ----------
    semivariogram_model : TheoreticalVariogram
        Fitted semivariogram.

    block_id : Union[str, Hashable]
        Block index.

    blocks_indexes : np.ndarray
        Indexes of other ps_blocks.

    is_directional : bool
        Is the semivariogram directional? Set to ``True`` if passed
        ``semivariogram_model`` has a direction.

    angular_tolerance : float, default = 1
        How many degrees of difference are allowed for the neighbors search.

    max_tick : int, default = 15
        How wide might the ``angular_tolerance`` (in one direction,
        in reality it is two times greater).

    point_support_distances : PointSupportDistance
        Object that calculates and stores distances between point supports
        of multiple ps_blocks.

    block_distances : numpy array
        Distances to the unknown block from other ps_blocks.

    block_angle_differences : numpy array, optional
        Angles between the neighboring ps_blocks and the unknown block. Estimated
        only when the semivariogram is directional.

    block_ds : DataFrame
        DataFrame for Poisson Kriging, with columns representing:
        other ps_blocks, points, values, distances, angles, core block index.

    unknown_n : int
        The number of unique point support samples in unknown area.

    Methods
    -------
    neighbors_angles : numpy array, property
        Get angles between the neighboring ps_blocks and the unknown block.

    neighbors_coordinates : numpy array, property
        Get coordinates of the neighboring ps_blocks.

    neighbors_distances : numpy array, property
        Get distances to the neighboring ps_blocks from the unknown block.

    kriging_input : DataFrame, property
        DataFrame representing unknown block's neighbors.

    neighbors_unique_indexes : numpy array, property
        Get indexes of the neighboring ps_blocks.

    pk_input : DataFrame, property
        DataFrame representing all ps_blocks and their relations to the unknown
        block. (See ``ds`` attribute).

    neighbors_values : numpy array, property
        Get values of the neighboring ps_blocks.

    blocks_dataframe : DataFrame, property
        DataFrame representing: other ps_blocks, points, values, distances,
        angles, core block index.

    point_to_block_ps_semivariance()
        Returns DataFrame with columns: block point support coordinate x,
        block point support coordinate y, neighbor index, average
        semivariance between coordinates and a neighbor

    point_to_block_ps_semivariance_array()
        Method calculates semivariances from each point of the block to
        other ps_blocks point support.

    distances_between_neighboring_point_supports()
        Parses distances between block neighbors.

    select_block_neighbors()
        Selects the closest neighbors.

    weighted_b2b_semivariance()
        Calculates weighted block to block semivariance.

    Raises
    ------
    AttributeError :
        - Angular differences were not provided and user wants to select
          neighbors based on direction.

    ValueError :
        - Kriging input is not calculated and user tries to access class
          properties.
        - For a given distance or angle no neighbors have been found.

    See Also
    --------
    select_poisson_kriging_data() :
        Function which uses this class to select neighbors for
        the interpolation.

    Examples
    --------
    >>> import os
    >>> import geopandas as gpd
    >>> from pyinterpolate import (
    >>>     Blocks,
    >>>     ExperimentalVariogram,
    >>>     PointSupport,
    >>>     TheoreticalVariogram
    >>> )
    >>> from pyinterpolate.core.data_models.poisson_kriging import PoissonKrigingInput
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
    >>> cpki = PoissonKrigingInput(
    ...     block_id=indexes[-5],
    ...     point_support=PS,
    ...     semivariogram_model=THEO
    ... )
    >>> cpki.select_block_neighbors(
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
                 no_closest_neighbors: int = 8,
                 blocks_indexes: np.ndarray = None):

        # TODO: store Block columns and PointSupport-Neighbors columns
        #  in separate dataclasses

        # Block columns
        self.blocks_indexes_column = 'blocks_indexes'
        self.distances_column = 'distances'
        self.angular_differences_column = 'angular_differences'
        self.block_id_column = 'block_id'
        self.block_points_column = 'points'
        self.block_total_point_support_column = 'total'
        self.block_values_column = 'values'

        # PointSupport - Neighbors columns
        self.ng_known_block_id_col = 'known_block_id'
        self.ng_kx_col = 'kx'
        self.ng_ky_col = 'ky'
        self.ng_kval_col = 'k_value'
        self.ng_ux_col = 'ux'
        self.ng_uy_col = 'uy'
        self.ng_uval_col = 'u_value'
        self.ng_distance_col = 'distance'
        self.ng_angular_difference_col = 'angular_difference'

        self.semivariogram_model = semivariogram_model

        self.block_id = block_id
        self.blocks_indexes = blocks_indexes
        self.__is_weighted = True  # always weighted
        self.is_directional = (
            True if self.semivariogram_model.direction is not None else False
        )
        self.angular_tolerance = None
        self.max_tick = None
        self.no_closest_neighbors = no_closest_neighbors
        self.point_support_distances = PointSupportDistance(
            verbose=False
        )  # todo set verbose

        # get distances and angles between ps_blocks
        self.block_distances = block_base_distances(
            block_id=self.block_id,
            point_support=point_support,
            point_support_distances=self.point_support_distances
        )

        self.block_angle_differences = None
        if self.is_directional:
            self.block_angle_differences = block_to_blocks_angles(
                block_id=self.block_id,
                point_support=point_support,
                direction=semivariogram_model.direction
            )

        # Set initial block dataset
        self.block_ds = set_blocks_dataset(
            block_id=self.block_id,
            points=point_support.blocks.block_representative_points,
            values=point_support.blocks.block_values,
            distances=self.block_distances,
            blocks_indexes=blocks_indexes,
            angular_differences=self.block_angle_differences,
            angular_differences_column=self.angular_differences_column,
            blocks_indexes_column=self.blocks_indexes_column,
            core_block_id_column=self.block_id_column,
            distances_column=self.distances_column,
            points_column=self.block_points_column,
            values_column=self.block_values_column
        )

        # Set additional consts
        self.unknown_n = self._unknown_no_points(block_id, point_support)

        # Neighbors finding parameters
        self._final_angular_tolerance = None
        self._from_any_point = False
        self._max_range = None

        # kriging input
        self._neighbors = None
        self._kriging_input = None

    @property
    def neighbors_coordinates(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, '
                             'please select neighbors first.')
        else:
            return self._kriging_input[[self.ng_kx_col, self.ng_ky_col]].to_numpy()

    @property
    def neighbors_distances(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, '
                             'please select neighbors first.')
        else:
            return self._kriging_input[self.ng_distance_col].to_numpy()

    @property
    def kriging_input(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, '
                             'please select neighbors first.')
        else:
            return self._kriging_input.copy()

    @property
    def neighbors_unique_indexes(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, '
                             'please select neighbors first.')
        else:
            return self._kriging_input[self.ng_known_block_id_col].unique()

    @property
    def blocks_dataframe(self):
        return self.block_ds.copy()

    @property
    def neighbors_values(self):
        if self._kriging_input is None:
            raise ValueError('Kriging input is not estimated yet, '
                             'please select neighbors first.')
        else:
            return self._kriging_input[self.ng_kval_col].to_numpy()

    def distances_between_neighboring_point_supports(
            self,
            point_support: PointSupport
    ) -> pd.DataFrame:
        """
        Function parses distances between block neighbors.

        Parameters
        ----------
        point_support : PointSupport
            Point Support representation with calculated distances
            between all points and all ps_blocks.

        Returns
        -------
        : pandas DataFrame
        """

        neighbors = self.neighbors_unique_indexes

        ds = []
        nset = set()

        for neighbor_a in neighbors:
            for neighbor_b in neighbors:

                nn = (neighbor_a, neighbor_b)
                r_nn = (neighbor_b, neighbor_a)

                if nn not in nset:
                    vals = self.point_support_distances.calc_pair_distances(
                        point_support=point_support,
                        block_pair=nn
                    )

                    pdf = pd.DataFrame(data=vals, columns=['block_a_value',
                                                           'block_b_value',
                                                           'distance'])
                    pdf['blocks_pair'] = [nn] * len(pdf)
                    ds.append(pdf)

                    nset.add(
                        nn
                    )

                    if neighbor_a != neighbor_b:
                        rdf = pd.DataFrame(data=vals, columns=[
                            'block_a_value', 'block_b_value', 'distance'
                        ])
                        rdf['blocks_pair'] = [r_nn] * len(pdf)
                        ds.append(rdf)
                        nset.add(r_nn)

        df = pd.concat(ds, ignore_index=True, axis=0)

        return df

    def distances_within_unknown_block(
            self,
            point_support: PointSupport
    ) -> pd.DataFrame:
        """
        Method prepares distances between all points within
        the unknown block's point support.

        Parameters
        ----------
        point_support : PointSupport
            Point Support representation with calculated distances between
            all points and all ps_blocks.

        Returns
        -------
        pdf : DataFrame
            ``{['point_a_value', 'point_b_value', 'distance', 'block_id']}``
        """
        nn = (self.block_id, self.block_id)
        vals = self.point_support_distances.calc_pair_distances(
            point_support=point_support,
            block_pair=nn
        )
        pdf = pd.DataFrame(data=vals, columns=[
            'point_a_value', 'point_b_value', 'distance'
        ])
        pdf['block_id'] = self.block_id
        return pdf

    def point_to_block_ps_semivariance(self):
        """
        Method calculates semivariances from each point of the block to
        other ps_blocks point support. Semivariance is
        averaged over neighbor's point support.

        Returns
        -------
        p2b : DataFrame
            DataFrame with columns:
              * block point support coordinate x
              * block point support coordinate y
              * neighbor index
              * average semivariance with a neighbor

        Notes
        -----
        In the first step method creates DataFrame with this structure:
          * (unknown) block point support coordinate x
          * (unknown) block point support coordinate y
          * (unknown) block point support value
          * neighbor index
          * neighboring block point support value
          * distance to the point in neighboring block
        """
        ds = self.kriging_input[[
            self.ng_ux_col,
            self.ng_uy_col,
            self.ng_uval_col,
            self.ng_known_block_id_col,
            self.ng_kval_col,
            self.ng_distance_col
        ]].copy()

        p2b = calculate_average_p2b_semivariance(
            ds=ds,
            semivariogram_model=self.semivariogram_model,
            block_x_coo_col=self.ng_ux_col,
            block_y_coo_col=self.ng_uy_col,
            block_val_col=self.ng_uval_col,
            neighbor_idx_col=self.ng_known_block_id_col,
            neighbor_val_col=self.ng_kval_col,
            distance_col=self.ng_distance_col
        )
        return p2b

    def point_to_block_ps_semivariance_array(self):
        """
        Method calculates semivariances from each point of the block to
        other ps_blocks point support. Semivariance is
        averaged over neighbor's point support.

        Returns
        -------
        arr : numpy array
            Numpy array where each row represents a neighbor, and each
            column is a different point from unknown point support.
        """
        ds = self.point_to_block_ps_semivariance()
        n = len(self.neighbors_unique_indexes)
        u_n = self.unknown_n

        arr = ds.to_numpy().reshape((n, u_n))

        return arr

    def select_block_neighbors(self,
                               point_support: PointSupport,
                               max_range: float,
                               no_closest_neighbors: int = 8,
                               angular_tolerance: float = None,
                               max_tick: float = None,
                               any_point: bool = True):
        """
        Function selects the closest neighbors.

        Parameters
        ----------
        point_support : PointSupport

        max_range : float
            The maximum distance for the neighbors search.

        no_closest_neighbors : int
            The minimum number of neighboring areas.

        angular_tolerance : float, default = 1
            How many degrees of difference are allowed for the neighbors
            search.

        max_tick : float, default = 10
            How wide might the ``angular_tolerance``
            (in one direction, in reality it is two times greater, up to
            180 -> 360 degrees)

        any_point : bool, default = True
            If neighboring block is selected, any point from it might
            be used to calculate angular difference. Setting
            this parameter to ``False`` changes the algorithm behavior,
            and neighbor's representative point must be
            within the given angle.

        Returns
        -------
        : DataFrame
        """
        self._update_private_neighbors_search_params(
            max_range=max_range,
            no_closest_neighbors=no_closest_neighbors,
            angular_tolerance=angular_tolerance,
            max_tick=max_tick,
            from_any_point=any_point
        )

        self._neighbors = self._select_neighbors()
        self._kriging_input = self._prepare_kriging_input(self._neighbors,
                                                          point_support)
        return self._kriging_input

    def weighted_b2b_semivariance(self) -> pd.Series:
        r"""
        Calculates the average weighted block-to-block semivariance.

        Returns
        -------
        : pd.Series
            {block index: weighted semivariance}

        Notes
        -----

        Weighted semivariance is calculated as:

        (1)

        $$\gamma_{v_{i}, v_{j}}
            =
            \frac{1}{\sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'}} *
                \sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'} *
                \gamma(u_{s}, u_{s'})$$

        where:
        * $w_{ss'}$ - product of point-support weights from
          block a and block b.
        * $\gamma(u_{s}, u_{s'})$ - semivariance between point-supports of
          block a and block b.
        """

        ds = self.kriging_input[[self.ng_known_block_id_col,
                                 self.ng_kval_col,
                                 self.ng_uval_col,
                                 self.ng_distance_col]]

        outs = weighted_avg_point_support_semivariances(
            semivariogram_model=self.semivariogram_model,
            distances_between_neighboring_point_supports=ds,
            index_col=self.ng_known_block_id_col,
            val1_col=self.ng_kval_col,
            val2_col=self.ng_uval_col,
            dist_col=self.ng_distance_col
        )

        return outs

    def _prepare_kriging_input(self, ds, point_support) -> pd.DataFrame:
        """
        Prepares valid Poisson Kriging Input.

        # [unknown x, unknown y, unknown point support value,
        #  other x, other y, other point support value,
        #  distance to other,
        #  angular difference,
        #  block id]

        Parameters
        ----------
        ds : DataFrame
            Selected Blocks - neighbors.

        point_support : PointSupport

        Returns
        -------
        : DataFrame
        """

        # TODO: select only points that lay in the same direction
        valid_blocks = ds.index.to_list()
        unknown_block = self.block_id

        if unknown_block in valid_blocks:
            valid_blocks.remove(unknown_block)

        # x, y, value - unknown block
        unknown_block_point_support = point_support.get_points_array(
            block_id=unknown_block
        )

        # distances between all points from unknown block point support
        # and other ps_blocks point supports
        others_point_support = [
            point_support.get_points_array(
                block_id=_other
            ) for _other in valid_blocks
        ]

        # get distances between all points from unknown block and other ps_blocks
        distances_to_others = [
            point_distance(
                points=unknown_block_point_support[:, :-1],
                other=_other[:, :-1]
            ) for _other in others_point_support
        ]

        # angles to others
        if self.is_directional:
            angles_to_others = [
                calc_angles_between_points(
                    vec1=unknown_block_point_support[:, :-1],
                    vec2=_other[:, :-1],
                    flatten_output=False
                ) for _other in others_point_support
            ]
        else:
            angles_to_others = None

        kriging_input = parse_kriging_input(
            unknown_points_and_values=unknown_block_point_support,
            known_blocks_id=valid_blocks,
            known_points_and_values=others_point_support,
            distances=distances_to_others,
            angle_diffs=angles_to_others,
        )

        return kriging_input

    def _select_neighbors(self):
        """
        Selects closest neighbors.

        Returns
        -------
        : numpy array
            Indexes of the closest neighbors.
        """
        neighbors_in_range = self._select_neighbors_in_range()

        # Check if direction is required
        if self.is_directional:
            neighbors_in_range = self._select_neighbors_in_direction(
                neighbors_in_range
            )

        return neighbors_in_range

    def _select_neighbors_in_direction(self, neighbors_in_range):
        if self.angular_tolerance is None:
            self.angular_tolerance = 1.0

        if self.max_tick is None:
            self.max_tick = 15.0

        angular_tolerance = self.angular_tolerance

        ads = neighbors_in_range[
            neighbors_in_range[
                self.angular_differences_column
            ] <= angular_tolerance
            ]

        while len(ads) < self.no_closest_neighbors:
            angular_tolerance = angular_tolerance + 1
            ads = neighbors_in_range[
                neighbors_in_range[
                    self.angular_differences_column
                ] <= angular_tolerance
            ]
            if self.max_tick <= angular_tolerance:
                break

        self._final_angular_tolerance = angular_tolerance

        if len(ads) == 0:
            raise ValueError('For a given distances and angle no neighbors '
                             'have been found. Make ``max_tick`` '
                             'larger or search for neighbors in '
                             'all directions.')
        else:
            return ads

    def _select_neighbors_in_range(self):
        """
        Function selects neighbors in the given range.

        Returns
        -------
        : DataFrame
        """
        ds = self.block_ds.copy(deep=True)

        ds = ds[ds[self.distances_column] > 0]
        ds.sort_values(by=self.distances_column, ascending=True, inplace=True)

        return ds.iloc[:self.no_closest_neighbors]

    @staticmethod
    def _unknown_no_points(block_id, point_support):
        """
        Method gets number of unique points in unknown area.

        Parameters
        ----------
        block_id : Union[str, Hashable]

        point_support : PointSupport

        Returns
        -------
        : int
        """
        arr = point_support.get_points_array(block_id=block_id)
        n = len(arr)
        return n

    def _update_private_neighbors_search_params(self,
                                                max_range,
                                                no_closest_neighbors,
                                                angular_tolerance,
                                                max_tick,
                                                from_any_point):
        self._max_range = max_range
        self.no_closest_neighbors = no_closest_neighbors
        self.angular_tolerance = angular_tolerance
        self.max_tick = max_tick
        self._from_any_point = from_any_point
