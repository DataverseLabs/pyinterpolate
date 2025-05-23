"""
Class to work with blocks & point support datasets and transformations.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

"""
from copy import deepcopy
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.point_support_distances import \
    PointSupportDistance
from pyinterpolate.semivariogram.deconvolution.avg_inblock_semivariance import \
    calculate_average_semivariance
from pyinterpolate.semivariogram.deconvolution.block_to_block_semivariance import \
    calculate_block_to_block_semivariance, \
    average_block_to_block_semivariances
from pyinterpolate.semivariogram.deconvolution.inblock import \
    calculate_inblock_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import \
    TheoreticalVariogram


class AggregatedVariogram:
    """
    Class calculates semivariances between blocks of aggregated data.

    Parameters
    ----------
    blocks : Blocks
        Aggregated data | choropleth map.

    point_support : PointSupport
        Point supports of ``blocks``.

    step_size : float
        Step size between lags.

    max_range : float
        Maximal distance of analysis.

    nugget : float, default = 0
        The nugget of blocks data.

    direction : float (in range [0, 360]), optional
        Direction of ``blocks`` semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

    variogram_weighting_method : str, default = "closest"
        Method used to weight error at a given lags. Available methods:

        - **equal**: no weighting,
        - **closest**: lags at a close range have bigger weights,
        - **distant**: lags that are further away have bigger custom_weights,
        - **dense**: error is weighted by the number of point pairs within
          a lag - more pairs, lower weight.

    models_group : str or list, default='safe'
        Models group to test:

        - 'all' - the same as list with all models,
        - 'safe' - ``['linear', 'power', 'spherical']``
        - as a list: multiple model types to test
        - as a single model type from the list:
            - 'circular',
            - 'cubic',
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical'.

    verbose : bool, default = False
        Print steps performed by the algorithm.

    log_process : bool, default = False
        Log process info (Level ``DEBUG``).

    Attributes
    ----------
    blocks : Blocks
        See ``aggregared_data`` parameter.

    step_size : float
        See ``step_size`` parameter.

    max_range : float
        See ``max_range`` parameter.

    agg_lags : numpy array
        Lags calculated as a set of equidistant points from ``step_size``
        to ``max_range`` with a step of size ``step_size``.

    tolerance : float, default = 1
        See ``tolerance`` parameter.

    direction : float, default = 0
        See ``direction`` parameter.

    nugget : float, default = 0
        See ``nugget`` parameter.

    models_group : str or list
        See ``models_group`` parameter.

    point_support : PointSupport
        See the ``point_support`` parameter.

    weighting_method : str, default = 'closest'
        See the ``variogram_weighting_method`` parameter.

    verbose : bool, default = False
        See ``verbose`` parameter.

    log_process : bool, default = False
        See the ``log_process`` parameter.

    experimental_variogram : ExperimentalVariogram
        The experimental variogram calculated from blocks (their coordinates).

    theoretical_model : TheoreticalVariogram
        The theoretical model derived from blocks' coordinates.

    inblock_semivariance : Dict
        ``{area id: the average inblock semivariance}``

    avg_inblock_semivariance : numpy array
        ``[lag, average inblocks semivariances, number of blocks within a lag]``

    block_to_block_semivariance : Dict
        ``{(block i, block j): [distance, semivariance, number of point pairs between blocks]}``

    avg_block_to_block_semivariance : numpy array
        ``[lag, semivariance, number of point pairs between blocks]``

    regularized_variogram : numpy array
        ``[lag, semivariance]``

    distances_between_blocks : Dict
        Weighted distances between all blocks:
        ``{block id : [distances to other blocks]}``


    Methods
    -------
    regularize()
        Method performs semivariogram regularization.

    show_semivariograms()
        Shows experimental variogram, theoretical model, average
        inblock semivariance, average block to block semivariance
        and regularized variogram.

    References
    ----------
    [1] Goovaerts P., Kriging and Semivariogram Deconvolution in
    the Presence of Irregular Geographical Units, Mathematical Geology 40(1),
    101-128, 2008

    """

    def __init__(self,
                 blocks: Blocks,
                 point_support: PointSupport,
                 step_size: float,
                 max_range: float,
                 nugget: float = 0,
                 direction: float = None,
                 tolerance: float = None,
                 variogram_weighting_method: str = 'closest',
                 models_group: Union[str, list] = 'safe',
                 verbose: bool = False,
                 log_process: bool = False):

        self.regularized_variogram = None
        self.blocks = blocks
        self.step_size = step_size
        self.max_range = max_range
        self.nugget = nugget
        self.agg_lags = np.arange(self.step_size,
                                  self.max_range,
                                  self.step_size)
        self.tolerance = tolerance
        self.direction = direction
        self.point_support = point_support
        self.point_support_distances = None
        self.weighting_method = variogram_weighting_method
        self.models_group = models_group

        # Process info
        self.verbose = verbose
        self.log_process = log_process

        # Variogram models
        #   from blocks
        self.experimental_variogram = None
        self.theoretical_model = None
        #  from point support within a single block
        self.inblock_semivariance = None
        # from inblock semivariances; lags -> from blocks
        self.avg_inblock_semivariance = None
        # from point supports between blocks
        self.block_to_block_semivariance = None
        # average from point supports between block pairs
        self.avg_block_to_block_semivariance = None

        # Model parameters
        self.distances_between_blocks = None

    def calculate_avg_inblock_semivariance(self) -> np.ndarray:
        r"""
        Method calculates the average semivariance within blocks
        :math:`\gamma_h(v, v)`. The average inblock semivariance is
        calculated as:

        .. math:: \gamma_h(v, v) = \frac{1}{(2*N(h))} \sum_{a=1}^{N(h)} [\gamma(v_{a}, v_{a}) + \gamma(v_{a+h}, v_{a+h})]

        where:

        - :math:`\gamma(v_{a}, v_{a})` is a semivariance within
          a block :math:`a`,
        - :math:`\gamma(v_{a+h}, v_{a+h})` is samivariance within a block at
          a distance :math:`h` from the block :math:`a`.

        Returns
        -------
        avg_inblock_semivariance : numpy array
             ``[lag, semivariance, number of block pairs]``

        """

        # Calculate inblock semivariance
        self.inblock_semivariance = calculate_inblock_semivariance(
            self.point_support,
            self.theoretical_model
        )

        # Calculate distances between blocks
        self.point_support_distances = PointSupportDistance(verbose=False)

        if self.distances_between_blocks is None:
            self.distances_between_blocks = self.point_support_distances.calculate_weighted_block_to_block_distances(
                point_support=self.point_support,
                return_distances=True
            )

        # Calc average semivariance
        avg_inblock_semivariance = calculate_average_semivariance(
            self.distances_between_blocks,
            self.inblock_semivariance,
            step_size=self.step_size,
            max_range=self.max_range
        )
        return avg_inblock_semivariance

    def calculate_avg_semivariance_between_blocks(self) -> np.ndarray:
        r"""
        Function calculates semivariance between areas based on their division
        into smaller blocks. It is :math:`\gamma(v, v_h)` - semivariogram value
        between any two blocks separated by the distance :math:`h`.

        Returns
        -------
        avg_block_to_block_semivariance : numpy array
            The average semivariance between the neighboring blocks
            point-supports:
            ``[lag, semivariance, number of block pairs within a range]``.

        Notes
        -----
        Block-to-block semivariance is calculated as:

        .. math:: \gamma(v_{a}, v_{a+h})=\frac{1} {P_{a}P_{a+h}}\sum_{s=1}^{P_{a}}\sum_{s'=1}^{P_{a+h}}\gamma(u_{s}, u_{s'})

        where:

        - :math:`\gamma(v_{a}, v_{a+h})` - block-to-block semivariance of
          the block :math:`a` and paired block :math:`a+h`.
        - :math:`P_{a}` and :math:`P_{a+h}` - number of support points within
          the block :math:`a` and block :math:`a+h`.
        - :math:`\gamma(u_{s}, u_{s'})` - semivariance of point supports
          between blocks.

        Then average block-to-block semivariance is calculated as:

        .. math:: \gamma_{h}(v, v_{h}) = \frac{1}{N(h)}\sum_{a=1}^{N(h)}\gamma(v_{a}, v_{a+h})

        where:

        - :math:`\gamma_{h}(v, v_{h})` - averaged block-to-block semivariances
          for a lag :math:`h`,
        - :math:`\gamma(v_{a}, v_{a+h})` - semivariance of block :math:`a`
          and paired block at a distance :math:`h`.
        """

        # Check if distances between blocks are calculated
        if self.distances_between_blocks is None:
            if self.point_support_distances.weighted_block_to_block_distances is None:
                self.point_support_distances.calculate_weighted_block_to_block_distances(
                    point_support=self.point_support,
                    return_distances=False
                )
                self.distances_between_blocks = self.point_support_distances.weighted_distances
            else:
                self.distances_between_blocks = self.point_support_distances.weighted_distances

        # {(block id a, block id b):
        # [distance, semivariance, number of point pairs between blocks]}
        self.block_to_block_semivariance = calculate_block_to_block_semivariance(
            self.point_support,
            self.distances_between_blocks,
            self.theoretical_model
        )

        semivars_arr = np.array(
            list(self.block_to_block_semivariance.values()),
            dtype=float
        )
        avg_block_to_block_semivariance = average_block_to_block_semivariances(
            semivariances_array=semivars_arr,
            lags=self.agg_lags
        )
        return avg_block_to_block_semivariance

    def regularize(self,
                   average_inblock_semivariances: np.ndarray = None,
                   semivariance_between_point_supports: np.ndarray = None,
                   experimental_block_semivariances: np.ndarray = None,
                   theoretical_block_model: TheoreticalVariogram = None
                   ) -> np.ndarray:
        r"""
        Method regularizes point support model. Procedure is described in [1].

        Parameters
        ----------
        average_inblock_semivariances : np.ndarray, default = None
            The mean semivariance between blocks. See Notes to learn more.

        semivariance_between_point_supports : np.ndarray, default = None
            Semivariance between all blocks calculated from
            the theoretical model.

        experimental_block_semivariances : np.ndarray, default = None
            The experimental semivariance between area coordinates.

        theoretical_block_model : TheoreticalVariogram, default = None
            A modeled variogram.

        Returns
        -------
        regularized_model : numpy array
            ``[lag, semivariance, number of point pairs, number of blocks included]``

        Notes
        -----
        Function has the form:

        .. math:: \gamma_v(h) = \gamma(v, v_h) - \gamma_h(v, v)

        where:

        - :math:`\gamma_v(h)` - the regularized variogram,
        - :math:`\gamma(v, v_h)` - a variogram value between any two blocks
          separated by the distance :math:`h` (calculated from their
          point support),
        - :math:`\gamma_h(v, v)` - average inblock semivariance between blocks.

        Average inblock semivariance between blocks:

        .. math:: \gamma_h(v, v) = \frac{1}{(2*N(h))} \sum_{a=1}^{N(h)} [\gamma(v_{a}, v_{a}) + \gamma(v_{a+h}, v_{a+h})]

        where :math:`\gamma(v_{a}, v_{a})` and
        :math:`\gamma(v_{a+h}, v_{a+h})` are inblock semivariances of block
        :math:`a` and block :math:`a+h` separated by the distance :math:`h`.

        References
        ----------
        [1] Goovaerts P., Kriging and Semivariogram Deconvolution in the
            Presence of Irregular Geographical Units,
            Mathematical Geology 40(1), 101-128, 2008
        """

        # Set all variograms and models
        if experimental_block_semivariances is None:
            self.experimental_variogram = self._get_experimental_variogram()
        else:
            self.experimental_variogram = experimental_block_semivariances

        if theoretical_block_model is None:
            self.theoretical_model = self._fit_theoretical_model()
        else:
            self.theoretical_model = theoretical_block_model

        # gamma_h(v, v)
        if average_inblock_semivariances is None:
            self.avg_inblock_semivariance = self.calculate_avg_inblock_semivariance()
        else:
            self.avg_inblock_semivariance = average_inblock_semivariances

        # gamma(v, v_h)
        if semivariance_between_point_supports is None:
            self.avg_block_to_block_semivariance = self.calculate_avg_semivariance_between_blocks()
        else:
            self.avg_block_to_block_semivariance = semivariance_between_point_supports

        # regularize variogram
        self.regularized_variogram = self.regularize_variogram()

        return self.regularized_variogram

    def regularize_variogram(self) -> np.ndarray:
        r"""
        Function regularizes semivariograms.

        Returns
        -------
        reg_variogram : numpy array
            ``[lag, semivariance]``

        Raises
        ------
        ValueError
            Semivariance at a given lag is below zero.

        Notes
        -----
        Regularized semivariogram is a difference between the average block
        to block semivariance :math:`\gamma(v, v_{h})` and the average inblock
        semivariances :math:`\gamma_{h}(v, v)` at a given lag :math:`h`.

        .. math:: \gamma_{v}(h)=\gamma(v, v_{h}) - \gamma_{h}(v, v)

        """

        reg_variogram = deepcopy(self.avg_block_to_block_semivariance[:,
                                 :-1])  # Get lag and semivariance
        reg_variogram[:, 1] = (
                self.avg_block_to_block_semivariance[:, 1] -
                self.avg_inblock_semivariance[:, 1]
        )

        if np.any(reg_variogram[:, 1] < 0):
            for row in reg_variogram:
                if row[1] < 0:
                    msg = (f'Calculated semivariance of regualrized variogram '
                           f'for the lag {row[0]} is below zero. '
                           f'Semivariance value: {row[1]}')
                    raise ValueError(msg)

        return reg_variogram

    def show_semivariograms(self):
        """
        Method plots:

        - experimental variogram,
        - theoretical model,
        - average inblock semivariance,
        - average block-to-block semivariance,
        - regularized variogram.

        Raises
        ------
        AttributeError
            The semivariogram regularization process has not been performed.
        """

        if self.regularized_variogram is None:
            raise AttributeError(
                'Variograms may be plot after regularization process. '
                'Use .regularize() method, then plot the results.'
            )
        else:
            plt.figure(figsize=(12, 6))
            # Plot experimental
            plt.scatter(self.agg_lags,
                        self.experimental_variogram.experimental_semivariances,
                        marker='8',
                        c='black')
            # Plot theoretical
            plt.plot(
                self.agg_lags,
                self.theoretical_model.fitted_model[:, 1],
                ':',
                color='black'
            )
            # Plot average inblock
            plt.plot(
                self.agg_lags,
                self.avg_inblock_semivariance[:, 1],
                '--',
                color='#e66101'
            )
            # Plot average block to block
            plt.plot(
                self.agg_lags,
                self.avg_block_to_block_semivariance[:, 1],
                '--',
                color='#fdb863'
            )
            # Plot regularized
            plt.plot(
                self.agg_lags,
                self.regularized_variogram[:, 1],
                color='#5e3c99'
            )
            legend = ['Experimental',
                      'Theoretical',
                      'Avg. Inblock',
                      'Avg. Block-to-block',
                      'Regularized']
            plt.legend(legend)
            plt.xlabel('Distance')
            plt.ylabel('Semivariance')
            plt.show()

    def _fit_theoretical_model(self) -> TheoreticalVariogram:
        """
        Method automatically fits experimental curve into the WEIGHTED
        theoretical model if it wasn't provided as a
        parameter in regularize() method.

        Returns
        -------
        theoretical_model : TheoreticalVariogram
            Weighted theoretical semivariogram.
        """
        theoretical_model = TheoreticalVariogram()
        theoretical_model.autofit(
            experimental_variogram=self.experimental_variogram,
            nugget=self.nugget,
            models_group=self.models_group,
            deviation_weighting=self.weighting_method
        )

        return theoretical_model

    def _get_experimental_variogram(self) -> ExperimentalVariogram:
        """
        Method gets experimental variogram from aggregated data.

        Returns
        -------
        experimental_variogram : ExperimentalVariogram
        """

        ds = self.blocks.representative_points_array()

        experimental_variogram = ExperimentalVariogram(
            ds=ds,
            step_size=self.step_size,
            max_range=self.max_range,
            direction=self.direction,
            tolerance=self.tolerance
        )

        return experimental_variogram


def regularize(blocks: Blocks,
               point_support: PointSupport,
               step_size: float,
               max_range: float,
               nugget: float = 0,
               direction: float = None,
               tolerance: float = None,
               average_inblock_semivariances: np.ndarray = None,
               semivariance_between_point_supports: np.ndarray = None,
               experimental_block_semivariances: np.ndarray = None,
               theoretical_block_model: TheoreticalVariogram = None,
               variogram_weighting_method: str = 'closest',
               verbose: bool = False,
               log_process: bool = False) -> np.ndarray:
    """
    Function is an alias for ``AggregatedVariogram`` class and performs
    semivariogram regularization. Function returns regularized variogram.

    Parameters
    ----------
    blocks : Blocks
        Aggregated data | choropleth map.

    point_support : PointSupport
            Point support of ``blocks``.

    step_size : float
        Step size between lags.

    max_range : float
        Maximal distance of analysis.

    nugget : float, default = 0
        The nugget of blocks data.

    direction : float (in range [0, 360]), optional, default=0
        Direction of ``blocks`` semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    average_inblock_semivariances : np.ndarray, default = None
        The mean semivariance between the blocks. See Notes to learn more.

    semivariance_between_point_supports : np.ndarray, default = None
        Semivariance between all blocks calculated from the theoretical model.

    experimental_block_semivariances : np.ndarray, default = None
        The experimental semivariance between area coordinates.

    theoretical_block_model : TheoreticalVariogram, default = None
        A modeled variogram.

    variogram_weighting_method : str, default = "closest"
        Method used to weight error at a given lags. Available methods:

        - **equal**: no weighting,
        - **closest**: lags at a close range have bigger custom_weights,
        - **distant**: lags that are further away have bigger custom_weights,
        - **dense**: error is weighted by the number of point pairs within a
          lag - more pairs, smaller weight.

    verbose : bool, default = False
        Print steps performed by the algorithm.

    log_process : bool, default = False
        Log process info (Level ``DEBUG``).

    Returns
    -------
    regularized : numpy array
        ``[lag, semivariance]``

    See Also
    --------
    AggregatedVariogram : core class for block semivariogram regularization

    References
    ----------
    [1] Goovaerts P., Kriging and Semivariogram Deconvolution in
    the Presence of Irregular Geographical
    Units, Mathematical Geology 40(1), 101-128, 2008
    """

    agg_var = AggregatedVariogram(
        blocks=blocks,
        point_support=point_support,
        step_size=step_size,
        max_range=max_range,
        direction=direction,
        tolerance=tolerance,
        nugget=nugget,
        variogram_weighting_method=variogram_weighting_method,
        verbose=verbose,
        log_process=log_process
    )

    regularized = agg_var.regularize(average_inblock_semivariances,
                                     semivariance_between_point_supports,
                                     experimental_block_semivariances,
                                     theoretical_block_model)
    return regularized
