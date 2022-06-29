from typing import Dict

import numpy as np

from pyinterpolate.distance.distance import calc_block_to_block_distance
from pyinterpolate.processing.polygon.structure import get_block_centroids_from_polyset
from pyinterpolate.variogram import TheoreticalVariogram, ExperimentalVariogram
from pyinterpolate.variogram.empirical import build_experimental_variogram
from pyinterpolate.variogram.regularization.block.avg_block_to_block_semivariances import \
    average_block_to_block_semivariances
from pyinterpolate.variogram.regularization.block.inblock_semivariance import calculate_inblock_semivariance
from pyinterpolate.variogram.regularization.block.avg_inblock_semivariances import calculate_average_semivariance
from pyinterpolate.variogram.regularization.block.block_to_block_semivariance import calculate_block_to_block_semivariance


class AggregatedVariogram:
    """
    Class calculates semivariance of aggregated counts.

    Parameters
    ----------
    aggregated_data : Dict
                      Dictionary retrieved from the PolygonDataClass, it's structure is defined as:

                          polyset = {
                                    'blocks': {
                                        'block index': {
                                            'value_name': float,
                                            'geometry_name': MultiPolygon | Polygon,
                                            'centroid.x': float,
                                            'centroid.y': float
                                        }
                                    }
                                    'info': {
                                            'index_name': the name of the index column,
                                            'geometry_name': the name of the geometry column,
                                            'value_name': the name of the value column,
                                            'crs': CRS of a dataset
                                    }
                                }

    agg_step_size : float
                    Step size between lags.

    agg_max_range : float
                    Maximal distance of analysis.

    point_support : Dict
                    Point support data as a Dict:

                        point_support = {
                          'area_id': [numpy array with points]
                        }

    agg_direction : float (in range [0, 360]), optional, default=0
                    direction of semivariogram, values from 0 to 360 degrees:
                    * 0 or 180: is NS direction,
                    * 90 or 270 is EW direction,
                    * 45 or 225 is NE-SW direction,
                    * 135 or 315 is NW-SE direction.

    agg_tolerance : float (in range [0, 1]), optional, default=1
                    If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                    the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                    the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                    for 0 tolerance.
                    * The minor axis size is (tolerance * step_size)
                    * The major axis size is ((1 - tolerance) * step_size)
                    * The baseline point is at a center of the ellipse.
                    Tolerance == 1 creates an omnidirectional semivariogram.

    variogram_weighting_method : str, default = "closest"
                                 Method used to weight error at a given lags. Available methods:
                                 - equal: no weighting,
                                 - closest: lags at a close range have bigger weights,
                                 - distant: lags that are further away have bigger weights,
                                 - dense: error is weighted by the number of point pairs within a lag - more pairs,
                                   lesser weight.

    verbose : bool, default = False
              Print steps performed by the algorithm.

    log_process : bool, default = False
                  Log process info (Level DEBUG).

    verbose : bool, default = False

    Attributes
    ----------

    """

    def __init__(self,
                 aggregated_data: Dict,
                 agg_step_size: float,
                 agg_max_range: float,
                 point_support: Dict,
                 agg_direction: float = 0,
                 agg_tolerance: float = 1,
                 variogram_weighting_method: str = 'closest',
                 verbose: bool = False,
                 log_process: bool = False):

        self.aggregated_data = aggregated_data
        self.agg_step_size = agg_step_size
        self.agg_max_range = agg_max_range
        self.agg_lags = np.arange(self.agg_step_size, self.agg_max_range, self.agg_step_size)
        self.agg_tolerance = agg_tolerance
        self.agg_direction = agg_direction
        self.point_support = point_support
        self.weighting_method = variogram_weighting_method

        # Process info
        self.verbose = verbose
        self.log_process = log_process

        # Variogram models
        self.experimental_variogram = None  # from blocks
        self.theoretical_model = None  # from blocks
        self.inblock_semivariance = None  # from point support within a block
        self.avg_inblock_semivariance = None  # from inblock semivariances; lags -> from blocks
        self.block_to_block_semivariances = None  # from point supports between blocks
        self.avg_block_to_block_semivariance = None  # average from point supports between block pairs
        self.regularized_variogram = None

        # Model parameters
        self.distances_between_blocks = None

    def regularize(self,
                   average_inblock_semivariances: np.ndarray = None,
                   semivariance_between_point_supports=None,
                   experimental_block_variogram=None,
                   theoretical_block_model=None) -> np.ndarray:
        """
        Method regularizes point support model. Procedure is described in [1].

        Parameters
        ----------
        average_inblock_semivariances : np.ndarray, optional
                                        The mean semivariance between the blocks. See Notes to learn more.

        semivariance_between_point_supports : np.ndarray, optional
                                              Semivariance between all blocks calculated from the theoretical model.

        experimental_block_variogram : np.ndarray, optional
                                       The experimental semivariance between area centroids.

        theoretical_block_model : TheoreticalVariogram, optional
                                  Modeled variogram.


        Returns
        -------
        regularized_model : numpy array
                            [lag, semivariance, number of point pairs]

        Notes
        -----
        Function has the form:

            gamma_v(h) = gamma(v, v_h) - gamma_h(v, v)

        where:
        - gamma_v(h) - regularized variogram,
        - gamma(v, v_h) - variogram value between any two blocks separated by the distance h (calculated from their
                          point support),
        - gamma_h(v, v) - average inblock semivariance between blocks.

        Average inblock semivariance between blocks:

            $$\gamma_h(v, v) = \frac{1}{(2*N(h))} \sum_{a=1}^{N(h)} [\gamma(v_{a}, v_{a}) + \gamma(v_{a+h}, v_{a+h})]$$

        where:
        - $\gamma(v_{a}, v_{a})$ and $\gamma(v_{a+h}, v_{a+h})$ are the inblock semivariances of block $a$ and
          block $a+h$ separated by the distance $h$.

        References
        ----------
        [1] Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units,
            Mathematical Geology 40(1), 101-128, 2008
        """
        # Set all variograms and models
        if experimental_block_variogram is None:
            self.experimental_variogram = self._get_experimental_variogram()

        if theoretical_block_model is None:
            self.theoretical_model = self._fit_theoretical_model()

        # gamma_h(v, v)
        if average_inblock_semivariances is None:
            self.avg_inblock_semivariance = self.calculate_avg_inblock_semivariance()

        # gamma(v, v_h)
        if semivariance_between_point_supports is None:
            self.avg_block_to_block_semivariance = self.calculate_avg_semivariance_between_blocks()

        # # Regularize
        # # gamma_v(h)
        # regularized_variogram = deepcopy(self.between_blocks_variogram)
        # regularized_variogram[:, 1] = (
        #     self.between_blocks_variogram[:, 1] - self.within_block_variogram[:, 1]
        # )
        #
        # if np.any(regularized_variogram[:, 1] < 0):
        #     raise ValueError('Calculated semivariances are below zero!')
        #
        # self.regularized_variogram = regularized_variogram.copy()

        return self.avg_block_to_block_semivariance

    def calculate_avg_inblock_semivariance(self) -> np.ndarray:
        """
        Method calculates the average semivariance within blocks gamma_h(v, v).

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        $$\gamma_h(v, v) = \frac{1}{(2*N(h))} \sum_{a=1}^{N(h)} [\gamma(v_{a}, v_{a}) + \gamma(v_{a+h}, v_{a+h})]$$

            where:
            - $\gamma(v_{a}, v_{a})$ is semivariance within a block $a$,
            - $\gamma(v_{a+h}, v_{a+h})$ is samivariance within a block at a distance $h$ from the block $a$.

        """

        # Calculate inblock semivariance
        if self.verbose:
            print('Start of the inblock semivariance calculation')

        # Pass dict with {area id, [points within area and their values]} and semivariogram model
        self.inblock_semivariance = calculate_inblock_semivariance(self.point_support, self.theoretical_model)

        if self.verbose:
            print('Inblock semivariance calculated successfully')

        # Calculate distances between blocks
        if self.verbose:
            print('Distances between blocks: calculation starts')

        self.distances_between_blocks = calc_block_to_block_distance(self.point_support)

        if self.verbose:
            print('Distances between blocks have been calculated')

        # Calc average semivariance
        avg_inblock_semivariance = calculate_average_semivariance(self.distances_between_blocks,
                                                                  self.inblock_semivariance,
                                                                  block_step_size=self.agg_step_size,
                                                                  block_max_range=self.agg_max_range)
        return avg_inblock_semivariance

    def calculate_avg_semivariance_between_blocks(self) -> np.ndarray:
        """
        Function calculates semivariance between areas based on their division into smaller blocks. It is
            gamma(v, v_h) - semivariogram value between any two blocks separated by the distance h.

        Returns
        -------
        avg_block_to_block_semivariance : numpy array
                                          The average semivariance between neighboring blocks point-supports:
                                          [lag, semivariance, number of block pairs within a range]
        Notes
        -----
        Block-to-block semivariance is calculated as:

        $$\gamma(v_{a}, v_{a+h})=\frac{1}{P_{a}P_{a+h}}\sum_{s=1}^{P_{a}}\sum_{s'=1}^{P_{a+h}}\gamma(u_{s}, u_{s'})$$

        where:
            - $\gamma(v_{a}, v_{a+h})$ - block-to-block semivariance of block $a$ and paired block $a+h$.
            - $P_{a}$ and $P_{a+h}$ - number of support points within block $a$ and block $a+h$.
            - $\gamma(u_{s}, u_{s'})$ - semivariance of point supports between blocks.

        Then average block-to-block semivariance is calculated as:

        $$\gamma_{h}(v, v_{h}) = \frac{1}{N(h)}\sum_{a=1}^{N(h)}\gamma(v_{a}, v_{a+h})$$

        where:
            - $\gamma_{h}(v, v_{h})$ - averaged block-to-block semivariances for a lag $h$,
            - $\gamma(v_{a}, v_{a+h})$ - semivariance of block $a$ and paired block at a distance $h$.
        """

        # Check if distances between blocks are calculated
        if self.distances_between_blocks is None:
            if self.verbose:
                print('Distances between blocks: calculation starts')

            self.distances_between_blocks = calc_block_to_block_distance(self.point_support)

            if self.verbose:
                print('Distances between blocks have been calculated')

        if self.verbose:
            print('Calculation of semivariances between areas separated by chosen lags')

        # {(block id a, block id b): [distance, semivariance, number of point pairs between blocks]}
        self.block_to_block_semivariances = calculate_block_to_block_semivariance(self.point_support,
                                                                                  self.distances_between_blocks,
                                                                                  self.theoretical_model)
        if self.verbose:
            print('Semivariance between blocks for a given lags calculated')
            print('Calculation of the mean semivariance for a given lag')

        semivars_arr = np.array(list(self.block_to_block_semivariances.values()), dtype=float)
        avg_block_to_block_semivariance = average_block_to_block_semivariances(semivariances_array=semivars_arr,
                                                                               lags=self.agg_lags,
                                                                               step_size=self.agg_step_size)
        return avg_block_to_block_semivariance

    def _fit_theoretical_model(self) -> TheoreticalVariogram:
        """
        Method automatically fits experimental curve into WEIGHTED theoretical model if it wasn't provided as a
        parameter in regularize() method.

        Returns
        -------
        theoretical_model : TheoreticalVariogram
                            Weighted theoretical model of semivariance.
        """
        theoretical_model = TheoreticalVariogram()
        theoretical_model.autofit(
            empirical_variogram=self.experimental_variogram,
            model_types='all',
            deviation_weighting=self.weighting_method
        )

        return theoretical_model

    def _get_experimental_variogram(self) -> ExperimentalVariogram:
        """
        Method gets experimental variogram from aggregated data if None is given in the regularize method.

        Returns
        -------
        gammas : ExperimentalVariogram
        """

        gammas = build_experimental_variogram(
            input_array=get_block_centroids_from_polyset(self.aggregated_data),
            step_size=self.agg_step_size,
            max_range=self.agg_max_range,
            weights=None,
            direction=self.agg_direction,
            tolerance=self.agg_tolerance
        )

        return gammas
