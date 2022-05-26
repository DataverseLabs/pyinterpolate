from copy import deepcopy

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

    variogram_weighting_method : str, default = "closest"
                                 Method used to weight error at a given lags. Available methods:
                                 - equal: no weighting,
                                 - closest: lags at a close range have bigger weights,
                                 - distant: lags that are further away have bigger weights,
                                 - dense: error is weighted by the number of point pairs within a lag - more pairs,
                                   lesser weight.

    verbose : bool, default = False

    Attributes
    ----------

    """

    def __init__(self,
                 aggregated_data: Dict,
                 agg_step_size: float,
                 agg_max_range: float,
                 point_support: Dict,
                 variogram_weighting_method: str = 'closest',
                 verbose: bool = False):

        self.aggregated_data = aggregated_data
        self.agg_step_size = agg_step_size
        self.agg_max_range = agg_max_range
        self.agg_tolerance = agg_tolerance
        self.agg_direction = agg_direction
        self.point_support = point_support
        self.weighting_method = variogram_weighting_method
        self.verbose = verbose

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

