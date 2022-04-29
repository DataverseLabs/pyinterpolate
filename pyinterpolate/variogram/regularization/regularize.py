from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from pyinterpolate.processing.structure import PolygonDataClass
from pyinterpolate.variogram import TheoreticalVariogram


class Deconvolution:
    """
    Class performs deconvolution of semivariogram of areal data. Whole procedure is based on the iterative process
    described in: [1].

    Steps to regularize semivariogram:
    - initialize your object (no parameters),
    - use fit() method to build initial point support model,
    - use transform() method to perform semivariogram regularization,
    - save semivariogram model with export_model() method.

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

    def __init__(self):
        pass

    def fit(self,
            agg_dataset: PolygonDataClass,
            point_support_dataset: np.ndarray,
            step_agg: float,
            max_range_agg: float,
            direction_agg: float = 0,
            tolerance_agg: float = 1,
            weight_lags_method: Union[str, int] = 0) -> None:
        # TODO weighting methods to Enum
        """

        Parameters
        ----------


        """

        # Update class parameters
        self.aggregated = agg_dataset
        self.point_support = point_support_dataset
        self.step_agg = step_agg
        self.max_range_agg = max_range_agg
        self.direction_agg = direction_agg
        self.tolerance_agg = tolerance_agg
        self.weight_lags_fn = self._select_weighting_method(weight_lags_method)


        pass

    def transform(self):
        pass

    def fit_transform(self):
        pass

    def import_model(self):
        pass

    def export_model(self):
        pass

    def plot(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass