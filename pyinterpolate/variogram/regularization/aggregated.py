from copy import deepcopy
from typing import Dict

import numpy as np

from pyinterpolate.variogram import TheoreticalVariogram
from pyinterpolate.variogram.empirical import calculate_semivariance


class AggVariogramPK:
    """
    Class calculates semivariance of aggregated counts.

    Parameters
    ----------
    aggregated_data : Dict
                      Dictionary retrieved from the PolygonDataClass, it's structure is defined as:

                          polyset = {
                            'points': numpy array with [centroid.x, centroid.y and value],
                            'igeom': list of [index, geometry polygon],
                            'info': {
                                'index_name': the name of the index column,
                                'geom_name': the name of the geometry column,
                                'val_name': the name of the value column,
                                'crs': CRS of a dataset
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
        self.agg_tolerance = agg_tolerance
        self.agg_direction = agg_direction
        self.point_support = point_support
        self.weighting_method = variogram_weighting_method

        # Process info
        self.verbose = verbose
        self.log_process = log_process

        # Variogram models
        self.experimental_variogram = None
        self.theoretical_model = None
        self.inblock_semivariance = None
        self.within_block_variogram = None
        self.between_blocks_variogram = None
        self.regularized_variogram = None

        # Model parameters
        self.distances_between_blocks = None

    def regularize(self,
                   within_block_variogram: np.ndarray = None,
                   between_blocks_variogram=None,
                   experimental_variogram=None,
                   theoretical_model=None) -> np.ndarray:
        """
        Method regularizes point support model. Procedure is described in [1].

        Parameters
        ----------
        within_block_variogram : np.ndarray, optional
                                 The mean variance between the blocks. See Notes to learn more.

        between_block_variogram : np.ndarray, optional
                                  Semivariance between all blocks calculated from the theoretical model.

        experimental_variogram : np.ndarray, optional
                              The experimental semivariance between area centroids.

        theoretical_model : TheoreticalVariogram, optional
                            Modeled variogram.


        Returns
        -------
        regularized_model : np.ndarray
                            [lag, semivariance, number of point pairs]

        Notes
        -----
        Function has the form:

            gamma_v(h) = gamma(v, v_h) - gamma_h(v, v)

        where:
        - gamma_v(h) - regularized variogram,
        - gamma(v, v_h) - variogram value between any two blocks separated by the distance h,
        - gamma_h(v, v) - arithmetical average of the within-block variogram.

        Within-block variogram definition:

            gamma_h(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)]

        where:
        - gamma(va, va) and gamma(va+h, va+h) are the inblock semivariances of block a and block a+h separated by
          the distance h weighted by the inblock sum of point support blocks.

        References
        ----------
        [1] Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units,
            Mathematical Geology 40(1), 101-128, 2008
        """
        # Set all variograms and models
        if experimental_variogram is None:
            experimental_variogram = self._get_experimental_variogram()
        self.experimental_variogram = experimental_variogram.copy()

        if theoretical_model is None:
            theoretical_model = self._fit_theoretical_model()
        self.theoretical_model = deepcopy(theoretical_model)

        # gamma_h(v, v)
        if within_block_variogram is None:
            within_block_variogram = self.calculate_mean_semivariance_within_blocks()
        self.within_block_variogram = within_block_variogram.copy()

        # gamma(v, v_h)
        if between_blocks_variogram is None:
            between_blocks_variogram = self.calculate_semivariance_between_blocks()
        self.between_blocks_variogram = between_blocks_variogram.copy()

        # Regularize
        # gamma_v(h)
        regularized_variogram = self.between_blocks_variogram.copy()
        regularized_variogram[:, 1] = (
            self.between_blocks_variogram[:, 1] - self.within_block_variogram[:, 1]
        )

        if np.any(regularized_variogram[:, 1] < 0):
            raise ValueError('Calculated semivariances are below zero!')

        self.regularized_variogram = regularized_variogram.copy()

        return self.regularized_variogram

    def calculate_mean_semivariance_within_blocks(self):
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

        # Pass numpy array with [area id, [points within area and their values]] and semivariogram model
        self.inblock_semivariance = calculate_inblock_semivariance(self.point_support, self.theoretical_model)

        if self.verbose:
            print('Inblock semivariance calculated successfully')
        #
        # # Calculate distance between blocks
        # if distances is None:
        #     if self.verbose:
        #         print('Distances between blocks: calculation starts')
        #     self.distances_between_blocks = calc_block_to_block_distance(self.within_area_points)
        #     if self.verbose:
        #         print('Distances between blocks have been calculated')
        # else:
        #     if self.verbose:
        #         print('Distances between blocks are provided, distance skipped, model parameters updated')
        #     self.distances_between_blocks = distances
        #
        # # Calc average semivariance
        # avg_semivariance = calculate_average_semivariance(self.distances_between_blocks, self.inblock_semivariance,
        #                                                   self.areal_lags, self.areal_ss)
        # return avg_semivariance


    def calculate_semivariance_between_blocks(self):
        pass

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



    def _get_experimental_variogram(self) -> np.ndarray:
        """
        Method gets experimental variogram from aggregated data if None is given in the regularize method.

        Returns
        -------
        gammas : np.ndarray
                 [lag, semivariance, number of points] from aggregated data
        """

        gammas = calculate_semivariance(
            points=self.aggregated_data['points'],
            step_size=self.agg_step_size,
            max_range=self.agg_max_range,
            weights=None,
            direction=self.agg_direction,
            tolerance=self.agg_tolerance
        )

        return gammas

