from typing import Dict

import numpy as np

from pyinterpolate.processing.checks import check_limits
from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.processing.polygon.structure import get_block_centroids_from_polyset, get_polyset_from_file
from pyinterpolate.variogram import build_experimental_variogram, TheoreticalVariogram
from pyinterpolate.variogram.regularization.aggregated import regularize


def calculate_deviation(theoretical: TheoreticalVariogram, regularized: np.ndarray) -> float:
    """
    Function calculates deviation between initial block variogram model and the regularized point support model.

    Parameters
    ----------
    theoretical : TheoreticalVariogram

    regularized : numpy array
                  [lag, semivariance]

    Returns
    -------
    deviation : float
                |Regularized - Theoretical| / Theoretical

    TODO: calculate deviation tests
    """
    lags = regularized[:, 0]
    reg_values = regularized[:, 1]
    theo_values = theoretical.predict(lags)
    numerator = np.abs(reg_values - theo_values)
    deviations = np.divide(numerator,
                          theo_values,
                          out=np.zeros_like(numerator),
                          where=theo_values != 0)
    deviation = float(np.mean(deviations))
    return deviation


class Deconvolution:
    """
    Class performs deconvolution of semivariogram of areal data. Whole procedure is based on the iterative process
    described in: [1].

    Steps to regularize semivariogram:
    - initialize your object (no parameters),
    - use fit() method to build initial point support model,
    - use transform() method to perform semivariogram regularization,
    - save deconvoluted semivariogram model with export() method.

    Attributes
    ----------
    experimental_variogram_areal : ExperimentalVariogram
                                   The experimental variogram of aggregated dataset.

    theoretical_variogram_areal : TheoreticalVariogram
                                  The modeled variogram of areal data.

    initial_reg_model : TODO

    theoretical_model : TheoreticalVariogram
                        TODO

    optimal_model : TODO

    areal_data : AggregatedDataClass
                 TODO

    point_support_data : numpy array
                         TODO


    weighting_method : str
                       How lags are weighted to calculate model error. Possible methods:
                       - TODO
                       - TODO

    store_models : bool
                   Should algorithm save a model from each iteration?

    weights : list
              Weights of each iteration.

    deviations : list
                 List of deviations for each iteration.

    iters_max : int
                A control parameter. Maximum number of iterations.

    iters_min : int
                A control parameter. Minimum number of iterations.

    deviation_ratio : float
                      A control parameter. Ratio of the initial regularization error and the last iteration
                      regularization error. Regularization error is the Mean Absolute Error between the regularized
                      areal semivariogram and the point-support theoretical semivariance. Smaller ratio > better model.

    min_deviation_ratio : float
                          A control parameter. The minimal deviation ratio when algorithm stops iterations.

    diff_decrease : float
                    A control parameter. Ratio of difference:
                    (The current error - The optimal model error) / (The optimal model error).
                    It is measured at each iteration.

    min_diff_decrease : float
                        A control parameter. The algorithm measures a relative error decrease in each iteration in
                        comparison to the optimal model. Usually, a tiny decrease for n_diffs times should
                        stop algorithm. We assume that model has reached its optimum.

    n_diffs : int
              A control parameter. Number of iterations when algorithm should stop if min_diff_decrease is low.

    Methods
    -------
    fit()
        Fits areal data and the point support data into a model, initializes the experimental semivariogram,
        the theoretical semivariogram model, regularized point support model, and deviation.

    transform()
        Performs semivariogram regularization.

    export_model()
        Exports regularized (or fitted) model.

    import_model()
        Imports regularized (or fitted) model.

    plot()
        Plots semivariances before and after regularization.


    References
    ----------
    [1] Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical
    Units, Mathematical Geology 40(1), 101-128, 2008

    Examples
    --------


    """

    def __init__(self, verbose=True):

        # Core data structures
        self.ps = None  # point support
        self.agg = None  # aggregated dataset

        # Initial variogram parameters
        self.agg_step = None
        self.agg_rng = None
        self.direction = None
        self.ranges = None
        self.tolerance = None
        self.weighting_method = None

        # Deviation and weights
        self.deviations = []
        self.initial_deviation = None
        self.weights = []

        # Variograms - initial
        self.initial_regularized_model = None
        self.initial_theoretical_agg_model = None
        self.initial_experimental_variogram = None

        # Variograms - temp

        # Variograms - optimal

        # Control
        self.verbose = verbose




    def fit(self,
            agg_dataset: Dict,
            point_support_dataset: Dict,
            agg_step_size: float,
            agg_max_range: float,
            agg_direction: float = 0,
            agg_tolerance: float = 1,
            variogram_weighting_method: str = "closest") -> None:
        """
        Function fits given areal data variogram into point support variogram - it is the first step of regularization
        process.

        Parameters
        ----------
        agg_dataset : Dict
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

        point_support_dataset : Dict
                                Point support data as a Dict:

                                    point_support = {
                                      'area_id': [numpy array with points]
                                    }

        agg_step_size : float
                        Step size between lags.

        agg_max_range : float
                        Maximal distance of analysis.

        agg_direction : float (in range [0, 360]), optional, default=0
                    direction of semivariogram, values from 0 to 360 degrees:
                    * 0 or 180: is NS direction,
                    * 90 or 270 is EW direction,
                    * 45 or 225 is NE-SW direction,
                    * 135 or 315 is NW-SE direction.

    agg_tolerance : float (in range [0, 1]), optional, default=1
                    If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                    the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0
                    then the bin is selected as an elliptical area with major axis pointed in the same direction as
                    the line for 0 tolerance.
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

        """

        if self.verbose:
            print('Regularization fit process starts')

        # Update class parameters
        self.agg = agg_dataset
        self.ps = point_support_dataset
        self.agg_step = agg_step_size
        self.agg_rng = agg_max_range
        self.ranges = np.arange(agg_step_size, agg_max_range, agg_step_size)
        self.direction = agg_direction
        self.tolerance = agg_tolerance
        self.weighting_method = variogram_weighting_method

        # Compute experimental variogram of areal data
        areal_centroids = get_block_centroids_from_polyset(self.agg)

        self.initial_experimental_variogram = build_experimental_variogram(
            input_array=areal_centroids,
            step_size=self.agg_step,
            max_range=self.agg_rng,
            direction=self.direction,
            tolerance=self.tolerance
        )

        # Compute theoretical variogram of areal data
        theo_model_agg = TheoreticalVariogram()
        theo_model_agg.autofit(
            self.initial_experimental_variogram,
            model_types='all'
        )
        self.initial_theoretical_agg_model = theo_model_agg

        # Regularize
        self.initial_regularized_model = regularize(
            aggregated_data=self.agg,
            agg_step_size=self.agg_step,
            agg_max_range=self.agg_rng,
            point_support=self.ps,
            agg_direction=self.direction,
            agg_tolerance=self.tolerance,
            variogram_weighting_method=self.weighting_method,
            verbose=True,
            log_process=False
        )

        self.initial_deviation = calculate_deviation(self.initial_theoretical_agg_model,
                                                     self.initial_regularized_model)

        self.deviations.append(self.initial_deviation)

        if self.verbose:
            print('Regularization fit process ends')

    def transform(self,
                  max_iters=25,
                  limit_deviation_ratio=0.01,
                  minimum_deviation_decrease=0.001,
                  reps_minimum_deviation_decrease=3):
        """
        Method performs semivariogram regularization after model fitting.

        Parameters
        ----------
        max_iters : int, default = 25
                    Maximum number of iterations.

        limit_deviation_ratio : float, default = 0.01
                                Minimal ratio of model deviation to initial deviation when algorithm is stopped.
                                Parameter must be set in the limits (0, 1).

        minimum_deviation_decrease : float, default = 0.001
                                     The minimum ratio of the difference between model deviation and optimal deviation
                                     to the optimal deviation: |dev - opt_dev| / opt_dev.
                                     Parameter must be set in the limits (0, 1).

        reps_minimum_deviation_decrease : int, default = 3
                                          How many consecutive repetitions of minimum_deviation_decrease must occur to
                                          stop the algorithm.

        Raises
        ------
        AttributeError : initial_regularized_model is undefined (user didn't perform fit() method).

        ValueError : limit_deviation_ratio or minimum_deviation_decrease parameters <= 0 or >= 1.

        """

        # Check if model was fitted
        self._check_fit()

        # Check limits
        check_limits(limit_deviation_ratio)
        check_limits(minimum_deviation_decrease)

        # Start processing




    def fit_transform(self):
        pass

    def import_model(self):
        pass

    def export_model(self):
        pass

    def plot(self):
        pass

    def _deviation(self):
        """
        Method calculates deviation between regularized model and experimental values.

        Returns
        -------

        """
        return -1

    def _check_fit(self):
        if self.initial_regularized_model is None:
            msg = 'The initial regularized model (initial_regularized_model attribute) is undefined. Perform fit()' \
                  'before transformation!'
            raise AttributeError(msg)

    # def _select_weighting_method(self, method_id: int) -> str:
    #     """
    #     Method returns weighting method used by the algorithm to calculate lags weights.
    #
    #     Parameters
    #     ----------
    #     method_id : int
    #                 How the error between the modeled variogram and real data is weighted with a distance?
    #                 - 0: no weighting,
    #                 - 1: lags at a close range have bigger weights,
    #                 - 2: lags at a large distance have bigger weights,
    #                 - 3: error is weighted by the number of point pairs within a lag.
    #
    #
    #     Returns
    #     -------
    #     weighting_function : str
    #         An alias to the function used to calculate deviation.
    #
    #     Raises
    #     ------
    #     KeyError
    #         ID is not defined (it is different than 0, 1, 2, 3)
    #
    #     """
    #
    #     if method_id == 0:
    #         return 'equal'
    #     elif method_id == 1:
    #         return 'closest'
    #     elif method_id == 2:
    #         return 'distant'
    #     elif method_id == 3:
    #         return 'dense'
    #     else:
    #         msg = 'Undefined id. You may select:\n' \
    #               '0: no weighting,\n' \
    #               '1: lags at a close range have bigger weights,\n' \
    #               '2: lags that are further away have bigger weights,\n' \
    #               '3: error is weighted by the number of point pairs within a lag.'
    #         raise KeyError(msg)

    def __str__(self):
        pass

    def __repr__(self):
        pass
