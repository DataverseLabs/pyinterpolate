"""
Semivariogram regularization & deconvolution process.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Union, List, Collection

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

from pyinterpolate.processing.checks import check_limits
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.transform.transform import get_areal_centroids_from_agg
from pyinterpolate.variogram import build_experimental_variogram, TheoreticalVariogram, ExperimentalVariogram
from pyinterpolate.variogram.regularization.aggregated import regularize


def calculate_deviation(theoretical: TheoreticalVariogram,
                        regularized: np.ndarray) -> float:
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
    - use ``fit()`` method to build initial point support model,
    - use ``transform()`` method to perform semivariogram regularization,
    - save deconvoluted semivariogram model with ``export_model()`` method.

    Attributes
    ----------
    ps : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]

        * Dict: ``{block id: [[point x, point y, value]]}``
        * numpy array: ``[[block id, x, y, value]]``
        * DataFrame and GeoDataFrame: columns = ``{x, y, ds, index}``
        * PointSupport

    agg : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
        Blocks with aggregated data.

        * Blocks: ``Blocks()`` class object.
        * GeoDataFrame and DataFrame must have columns: ``centroid.x, centroid.y, ds, index``.
          Geometry column with polygons is not used and optional.
        * numpy array: ``[[block index, centroid x, centroid y, value]]``.

    initial_regularized_variogram : numpy array
        ``[lag, semivariance]``

    initial_theoretical_agg_model : TheoreticalVariogram

    initial_theoretical_model_prediction : numpy array
        ``[lag, semivariance]``

    initial_experimental_variogram : numpy array
        ``[lag, semivariance, number of pairs]``

    final_theoretical_model : TheoreticalVariogram

    final_optimal_variogram : numpy array
        ``[lag, semivariance]``

    agg_step : float
        Step size between lags.

    agg_rng : float
        Maximal distance of analysis.

    ranges : numpy array
        ``np.arange(agg_step, agg_rng, agg_step)``

    direction : float (in range [0, 360])
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1])

        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y axis and direction parameter. If ``tolerance`` is ``> 0`` then
        the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    weighting_method : str
        Method used to weight error at a given lags. Available methods:

        - **equal**: no weighting,
        - **closest**: lags at a close range have bigger weights,
        - **distant**: lags that are further away have bigger weights,
        - **dense**: error is weighted by the number of point pairs within a lag - more pairs, lesser weight.

    deviations : List
        List of deviations per iteration. The first element is the initial deviation.

    weights : List
        List of weights applied to lags in each iteration.

    verbose : bool
        Should algorithm ``print()`` process steps into a terminal.

    store_models : bool
        Should theoretical and regularized models be stored after each iteration.

    theoretical_models : List
        List with theoretical models parameters.

    regularized_models : List
        List with numpy arrays with regularized models.


    Methods
    -------
    fit()
        Fits areal data and the point support data into a model, initializes the experimental semivariogram,
        the theoretical semivariogram model, regularized point support model, and deviation.

    transform()
        Performs semivariogram regularization.

    fit_transform()
        Performs fit() and transform() at one time.

    export_model()
        Exports regularized (or fitted) model.

    plot_variograms()
        Plots semivariances before and after regularization.

    plot_deviations()
        Plots each deviation divided by the initial deviation.

    plot_weights()
        Plots the mean weight value per lag.

    References
    ----------
    [1] Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical
    Units, Mathematical Geology 40(1), 101-128, 2008

    Examples
    --------
    >>> dcv = Deconvolution(verbose=True)
    >>> dcv.fit(agg_dataset=...,
    ...         point_support_dataset=...,
    ...         agg_step_size=...,
    ...         agg_max_range=...,
    ...         variogram_weighting_method='closest')
    >>> dcv.transform(max_iters=5)
    >>> dcv.plot_variograms()
    >>> dcv.plot_deviations()
    >>> dcv.plot_weights()
    >>> dcv.export_model('results.csv')
    """

    def __init__(self, verbose=True, store_models=False):

        # Core data structures
        self.ps = None  # point support
        self.agg = None  # aggregated dataset

        # Initial variogram parameters
        self.agg_step = None
        self.agg_rng = None
        self.agg_nugget = None
        self.direction = None
        self.ranges = None
        self.tolerance = None
        self.weighting_method = None
        self.model_types = None

        # Deviation and weights
        self.deviations = []
        self.initial_deviation = None
        self.optimal_deviation = None
        self.weights = []

        # Variograms - initial
        self.initial_regularized_variogram = None
        self.initial_theoretical_agg_model = None
        self.initial_theoretical_model_prediction = None
        self.initial_experimental_variogram = None

        # Variograms - optimal
        self.s2 = None  # sill of initial theoretical model squared
        self.optimal_theoretical_model = None
        self.optimal_regularized_variogram = None

        # Variograms - final
        self.final_theoretical_model = None
        self.final_optimal_variogram = None

        # Control
        self.verbose = verbose
        self.store_models = store_models
        self.iter = 0
        self.max_iters = 0
        self.min_deviation_ratio = None
        self.min_deviation_decrease = None
        self.deviation_counter = 0
        self.reps_deviation_decrease = 0
        self.w_change = False
        self.was_fit = False
        self.was_transformed = False

        # Debug and stability
        self.theoretical_models = []  # List with theoretical models parameters
        self.regularized_models = []  # List with numpy arrays with regularized models

    def fit(self,
            agg_dataset: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
            point_support_dataset: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
            agg_step_size: float,
            agg_max_range: float,
            agg_nugget: float = 0,
            agg_direction: float = None,
            agg_tolerance: float = 1,
            variogram_weighting_method: str = "closest",
            model_types: Union[str, List] = 'basic') -> None:
        """
        Function fits given areal data variogram into point support variogram - it is the first step of regularization
        process.

        Parameters
        ----------
        agg_dataset : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
            Blocks with aggregated data.

            * Blocks: ``Blocks()`` class object.
            * GeoDataFrame and DataFrame must have columns: ``centroid.x, centroid.y, ds, index``.
              Geometry column with polygons is not used.
            * numpy array: ``[[block index, centroid x, centroid y, value]]``.

        point_support_dataset : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]

            * Dict: ``{block id: [[point x, point y, value]]}``
            * numpy array: ``[[block id, x, y, value]]``
            * DataFrame and GeoDataFrame: columns = ``{x, y, ds, index}``
            * PointSupport

        agg_step_size : float
            Step size between lags.

        agg_max_range : float
            Maximal distance of analysis.

        agg_nugget : float, default = 0
            The nugget of a data.

        agg_direction : float (in range [0, 360]), optional, default=0
            Direction of semivariogram, values from 0 to 360 degrees:

            - 0 or 180: is E-W,
            - 90 or 270 is N-S,
            - 45 or 225 is NE-SW,
            - 135 or 315 is NW-SE.

        agg_tolerance : float (in range [0, 1]), optional, default=1
            If ``agg_tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
            the coordinate system and the direction given by y axis and direction parameter. If ``agg_tolerance`` is ``> 0``
            then the bin is selected as an elliptical area with major axis pointed in the same direction as the line
            for 0 tolerance.

            * The major axis size == ``agg_step_size``.
            * The minor axis size is ``agg_tolerance * agg_step_size``
            * The baseline point is at a center of the ellipse.
            * ``agg_tolerance == 1`` creates an omnidirectional semivariogram.

        variogram_weighting_method : str, default = "closest"
            Method used to weight error at a given lags. Available methods:

            - **equal**: no weighting,
            - **closest**: lags at a close range have bigger weights,
            - **distant**: lags that are further away have bigger weights,
            - **dense**: error is weighted by the number of point pairs within a lag - more pairs, lesser weight.

        model_types : str or List, default='basic'
            List of modeling functions or a name of a single function. Available models:

            - 'all' - the same as list with all models,
            - 'basic' - ['exponential', 'linear', 'power', 'spherical'],
            - 'circular',
            - 'cubic',
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical',
            - or a different set of the above.
        """

        if self.verbose:
            print('Regularization fit process starts')

        # Update class parameters
        self.agg = agg_dataset
        self.ps = point_support_dataset
        self.agg_step = agg_step_size
        self.agg_rng = agg_max_range
        self.agg_nugget = agg_nugget
        self.ranges = np.arange(agg_step_size, agg_max_range, agg_step_size)
        self.direction = agg_direction
        self.tolerance = agg_tolerance
        self.weighting_method = variogram_weighting_method
        self.model_types = self._parse_model_types(model_types)

        # Compute experimental variogram of areal data
        areal_centroids = get_areal_centroids_from_agg(self.agg)

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
            nugget=self.agg_nugget,
            model_types=self.model_types,
            deviation_weighting=self.weighting_method
        )
        self.initial_theoretical_agg_model = theo_model_agg
        self.s2 = self.initial_theoretical_agg_model.sill
        self.initial_theoretical_model_prediction = self.initial_theoretical_agg_model.fitted_model[:, 1]
        # self.initial_theoretical_model_prediction = self.initial_theoretical_agg_model.predict(
        #     self.initial_theoretical_agg_model.lags
        # )

        # Regularize
        self.initial_regularized_variogram = regularize(
            aggregated_data=self.agg,
            agg_step_size=self.agg_step,
            agg_max_range=self.agg_rng,
            agg_nugget=self.agg_nugget,
            point_support=self.ps,
            theoretical_block_model=self.initial_theoretical_agg_model,
            experimental_block_variogram=self.initial_experimental_variogram.experimental_semivariance_array,
            agg_direction=self.direction,
            agg_tolerance=self.tolerance,
            variogram_weighting_method=self.weighting_method,
            verbose=True,
            log_process=False
        )

        self.initial_deviation = calculate_deviation(self.initial_theoretical_agg_model,
                                                     self.initial_regularized_variogram)

        self.deviations.append(self.initial_deviation)

        self.was_fit = True
        self.iter = 1

        if self.verbose:
            print('Regularization fit process ends')

    def transform(self,
                  max_iters=25,
                  limit_deviation_ratio=0.1,
                  minimum_deviation_decrease=0.01,
                  reps_deviation_decrease=3):
        """
        Method performs semivariogram regularization after model fitting.

        Parameters
        ----------
        max_iters : int, default = 25
            Maximum number of iterations.

        limit_deviation_ratio : float, default = 0.1
            Minimal ratio of model deviation to initial deviation when algorithm is stopped.
            Parameter must be set within the limits ``(0, 1)``.

        minimum_deviation_decrease : float, default = 0.01
            The minimum ratio of the difference between model deviation and optimal deviation
            to the optimal deviation: ``|dev - opt_dev| / opt_dev``.
            Parameter must be set within the limits ``(0, 1)``.

        reps_deviation_decrease : int, default = 3
            How many repetitions of small deviation decrease before termination of the algorithm.

        Raises
        ------
        AttributeError
            ``initial_regularized_model`` is undefined (user didn't perform ``fit()`` method).

        ValueError
            ``limit_deviation_ratio`` or ``minimum_deviation_decrease`` parameters ``<= 0 or >= 1``.

        """

        print('Transform procedure starts')

        # Check if model was fitted
        self._check_fit()

        # Check limits
        check_limits(limit_deviation_ratio)
        check_limits(minimum_deviation_decrease)

        # Update optimal models with initial models - make copies to be sure that we didn't overwrite values
        self.max_iters = max_iters
        self.min_deviation_ratio = limit_deviation_ratio
        self.min_deviation_decrease = minimum_deviation_decrease
        self.reps_deviation_decrease = reps_deviation_decrease

        initial_model_params = self.initial_theoretical_agg_model.to_dict()
        self.optimal_theoretical_model = self.initial_theoretical_agg_model
        self.optimal_regularized_variogram = self.initial_regularized_variogram.copy()
        self.optimal_deviation = self.initial_deviation

        # Append models if store_models parameter is set to True
        if self.store_models:
            self.theoretical_models.append(initial_model_params)
            self.regularized_models.append(self.initial_regularized_variogram)

        # Start iterative procedure
        for i in trange(self.max_iters):
            deviation_test = self._check_transform(i)
            if deviation_test:

                if self.verbose:
                    print('Process terminated: deviation gain is too small')

                break
            else:
                # Compute new experimental values for new experimental point support model
                rescaled_experimental_variogram = self._rescale_optimal_theoretical_model()

                # Fit rescaled model to the new theoretical fn
                temp_theoretical_semivariogram_model = TheoreticalVariogram()
                temp_theoretical_semivariogram_model.autofit(
                    self._rescaled_to_exp_variogram(rescaled_experimental_variogram),
                    model_types=self.model_types,
                    rang=self.initial_theoretical_agg_model.rang,
                    nugget=self.agg_nugget,
                    deviation_weighting=self.weighting_method
                )

                # Regularize model
                temp_regularized_variogram = regularize(
                    aggregated_data=self.agg,
                    agg_step_size=self.agg_step,
                    agg_max_range=self.agg_rng,
                    agg_nugget=self.agg_nugget,
                    point_support=self.ps,
                    theoretical_block_model=temp_theoretical_semivariogram_model,
                    experimental_block_variogram=rescaled_experimental_variogram,
                    agg_direction=self.direction,
                    agg_tolerance=self.tolerance,
                    variogram_weighting_method=self.weighting_method,
                    verbose=True,
                    log_process=False
                )

                # Compute diff stats
                current_deviation = calculate_deviation(self.initial_theoretical_agg_model,
                                                        temp_regularized_variogram)

                if current_deviation < self.optimal_deviation:
                    self.w_change = False
                    self.optimal_deviation = current_deviation

                    self.optimal_theoretical_model = temp_theoretical_semivariogram_model
                    self.optimal_regularized_variogram = temp_regularized_variogram

                else:
                    self.w_change = True

                self.deviations.append(current_deviation)
                self.iter = self.iter + 1

                # Append models if store_models parameter is set to True
                if self.store_models:
                    self.theoretical_models.append(temp_theoretical_semivariogram_model)
                    self.regularized_models.append(temp_regularized_variogram)

        # Get theoretical model from regularized
        self.final_theoretical_model = self.optimal_theoretical_model
        self.final_optimal_variogram = self.optimal_regularized_variogram

        self.was_transformed = True

    def fit_transform(self,
                      agg_dataset: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                      point_support_dataset: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                      agg_step_size: float,
                      agg_max_range: float,
                      agg_nugget: float = 0,
                      agg_direction: float = None,
                      agg_tolerance: float = 1,
                      variogram_weighting_method: str = "closest",
                      model_types: Union[str, List] = 'basic',
                      max_iters=25,
                      limit_deviation_ratio=0.1,
                      minimum_deviation_decrease=0.01,
                      reps_deviation_decrease=3):
        """
        Method performs ``fit()`` and ``transform()`` operations at once.

        Parameters
        ----------
        agg_dataset : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
            Blocks with aggregated data.

            * Blocks: ``Blocks()`` class object.
            * GeoDataFrame and DataFrame must have columns: ``centroid.x, centroid.y, ds, index``.
              Geometry column with polygons is not used.
            * numpy array: ``[[block index, centroid x, centroid y, value]]``.

        point_support_dataset : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]

            * Dict: ``{block id: [[point x, point y, value]]}``
            * numpy array: ``[[block id, x, y, value]]``
            * DataFrame and GeoDataFrame: columns = ``{x, y, ds, index}``
            * PointSupport

        agg_step_size : float
            Step size between lags.

        agg_max_range : float
            Maximal distance of analysis.

        agg_nugget : float, default = 0
            The nugget of a dataset.

        agg_direction : float (in range [0, 360]), default=0
            Direction of semivariogram, values from 0 to 360 degrees:

            - 0 or 180: is E-W,
            - 90 or 270 is N-S,
            - 45 or 225 is NE-SW,
            - 135 or 315 is NW-SE.

        agg_tolerance : float (in range [0, 1]), optional, default=1
            If ``agg_tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
            the coordinate system and the direction given by y axis and direction parameter. If ``agg_tolerance`` is ``> 0``
            then the bin is selected as an elliptical area with major axis pointed in the same direction as the line
            for 0 tolerance.

            * The major axis size == ``agg_step_size``.
            * The minor axis size is ``agg_tolerance * agg_step_size``
            * The baseline point is at a center of the ellipse.
            * ``agg_tolerance == 1`` creates an omnidirectional semivariogram.

        variogram_weighting_method : str, default = "closest"
            Method used to weight error at a given lags. Available methods:

            - **equal**: no weighting,
            - **closest**: lags at a close range have bigger weights,
            - **distant**: lags that are further away have bigger weights,
            - **dense**: error is weighted by the number of point pairs within a lag - more pairs, lesser weight.

        model_types : str or List, default='basic'
            List of modeling functions or a name of a single function. Available models:

            - 'all' - the same as list with all models,
            - 'basic' - ['exponential', 'linear', 'power', 'spherical'],
            - 'circular',
            - 'cubic',
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical',
            - or a different set of the above.

        max_iters : int, default = 25
            Maximum number of iterations.

        limit_deviation_ratio : float, default = 0.01
            Minimal ratio of model deviation to initial deviation when algorithm is stopped. Parameter must be set
            within the limits ``(0, 1)``.

        minimum_deviation_decrease : float, default = 0.001
            The minimum ratio of the difference between model deviation and optimal deviation to the optimal
            deviation: ``|dev - opt_dev| / opt_dev``. The parameter must be set within the limits ``(0, 1)``.

        reps_deviation_decrease : int, default = 3
            How many repetitions of small deviation decrease before termination of the algorithm.
        """

        self.fit(agg_dataset=agg_dataset,
                 point_support_dataset=point_support_dataset,
                 agg_step_size=agg_step_size,
                 agg_max_range=agg_max_range,
                 agg_nugget=agg_nugget,
                 agg_direction=agg_direction,
                 agg_tolerance=agg_tolerance,
                 variogram_weighting_method=variogram_weighting_method,
                 model_types=model_types)

        self.transform(max_iters=max_iters,
                       limit_deviation_ratio=limit_deviation_ratio,
                       minimum_deviation_decrease=minimum_deviation_decrease,
                       reps_deviation_decrease=reps_deviation_decrease)

    def export_model(self, fname: str):
        """
        Function exports final theoretical model.

        Parameters
        ----------
        fname : str
            File name for model parameters (nugget, sill, range, model type)

        Raises
        ------
        RunetimeError
            A model hasn't been transformed yet.
        """

        if self.final_theoretical_model is None:
            raise RuntimeError('You cannot export any model if you not transform data.')

        self.final_theoretical_model.to_json(fname)

    def plot_variograms(self):
        """
        Function shows experimental semivariogram, theoretical semivariogram and regularized semivariogram after
        semivariogram regularization with ``transform()`` method.
        """
        lags = self.initial_experimental_variogram.lags

        plt.figure(figsize=(12, 6))
        plt.plot(lags,
                 self.initial_experimental_variogram.experimental_semivariances, 'bo')
        plt.plot(lags,
                 self.initial_theoretical_agg_model.predict(lags), color='r',
                 linestyle='--')

        if self.final_optimal_variogram is not None:
            plt.plot(lags, self.final_optimal_variogram[:, 1], 'go')

            plt.plot(lags,
                     self.final_theoretical_model.predict(lags), color='black', linestyle='dotted')
            plt.legend(['Experimental semivariogram of areal data', 'Initial Semivariogram of areal data',
                        'Regularized data points, iteration {}'.format(self.iter),
                        'Optimized theoretical point support model'])
            plt.title('Semivariograms comparison. Deviation value: {}'.format(self.optimal_deviation))
        else:
            plt.legend(['Experimental semivariogram of areal data', 'Initial Semivariogram of areal data'])
            plt.title('Semivariograms comparison')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    def plot_deviations(self):
        plt.figure(figsize=(12, 6))
        plt.plot(
            np.arange(len(self.deviations)),
            [x / self.initial_deviation for x in self.deviations]
        )
        plt.xlabel('Iteration')
        plt.ylabel('Deviation')
        plt.show()

    def plot_weights(self):
        plt.figure(figsize=(12, 6))
        plt.plot(
            np.arange(len(self.weights)),
            [np.mean(weight) for weight in self.weights]
        )
        plt.xlabel('Iteration')
        plt.ylabel('Average weight')
        plt.show()

    def _check_fit(self):
        if self.initial_regularized_variogram is None:
            msg = 'The initial regularized model (initial_regularized_model attribute) is undefined. Perform fit()' \
                  'before transformation!'
            raise AttributeError(msg)

    def _check_transform(self, iter_no: int):
        # Test deviation ratio
        if self._deviation_ratio():
            return True

        # Test deviation decrease
        if self._deviation_decrease(iter_no):
            return True

        return False

    def _deviation_decrease(self, iter_no: int) -> bool:
        """
        |dev - opt_dev| / opt_dev

        Parameters
        ----------
        iter_no : int

        Returns
        -------
        : bool
        """
        if iter_no == 0:
            return False
        else:
            dev_decrease = (self.deviations[-1] - self.optimal_deviation) / self.optimal_deviation

            if dev_decrease < 0 and abs(dev_decrease) <= self.min_deviation_decrease:
                self.deviation_counter = self.deviation_counter + 1
                if self.deviation_counter == self.reps_deviation_decrease:
                    return True
                return False
            else:
                self.deviation_counter = 0
                return False

    def _deviation_ratio(self) -> bool:
        """
        The model deviation to initial deviation.

        Returns
        -------
        : bool
        """
        dev_ratio = self.deviations[-1] / self.initial_deviation
        if dev_ratio <= self.min_deviation_ratio:
            return True
        return False

    @staticmethod
    def _parse_model_types(model_types):
        """
        The first level check and parser for model types.

        Parameters
        ----------
        model_types : List or str

        Returns
        -------
        mtypes : List
        """

        all_models = [
                    'circular',
                    'cubic',
                    'exponential',
                    'gaussian',
                    'linear',
                    'power',
                    'spherical'
        ]

        basic_models = [
            'exponential',
            'linear',
            'power',
            'spherical'
        ]

        if isinstance(model_types, str):
            if model_types == 'all':
                return all_models
            elif model_types == 'basic':
                return basic_models
            else:
                return [model_types]

        elif isinstance(model_types, Collection):
            return model_types

        else:
            raise TypeError('Unknown Type of the input, model_types parameter takes str or List as an input.')

    def _rescale_optimal_theoretical_model(self) -> np.ndarray:
        """
        Function rescales points derived from the optimal theoretical model and creates new experimental
        values based on the equation:

        $$\gamma_{res}(h) = \gamma_{opt}(h) \times w(h)$$

        $$w(h) = w(h) = 1 + \frac{\gamma_{v}^{exp}(h)-\gamma_{v}^{opt}}{s^{2}\sqrt{iter}}$$

        where:

        - $\gamma_{v}^{exp}(h)$ : theoretical model fitted to blocks (1st iter), then point support model that has been
                                  derived from a rescaled values.
        - $\gamma_{v}^{opt}$ : optimal point support model (after regularization),
        - $w(h)$ : weights vector (each record is a weight applied to a specific lag),
        - $s$ - sill of the theoretical model fitted to the blocks,
        - $iter$ - iteration number.

        Returns
        -------
        rescaled : numpy array
                    Rescalled point support model.

        """

        y_opt_h = self.optimal_theoretical_model.predict(self.ranges)

        if not self.w_change:
            denom = self.s2 * np.sqrt(self.iter)
            numer = self.initial_theoretical_model_prediction - self.optimal_regularized_variogram[:, 1]
            w = 1 + (numer / denom)
        else:
            w = 1 + (self.weights[-1] - 1) / 2

        rescaled = np.zeros_like(self.optimal_regularized_variogram)
        rescaled[:, 0] = self.optimal_regularized_variogram[:, 0]
        rescaled[:, 1] = y_opt_h * w

        self.weights.append(w)

        return rescaled

    def _rescaled_to_exp_variogram(self, rescaled: np.ndarray) -> ExperimentalVariogram:
        exp_var = ExperimentalVariogram(input_array=self.initial_experimental_variogram.input_array,
                                        step_size=self.initial_experimental_variogram.step,
                                        max_range=self.initial_experimental_variogram.mx_rng,
                                        weights=self.initial_experimental_variogram.weights,
                                        direction=self.initial_experimental_variogram.direct,
                                        tolerance=self.initial_experimental_variogram.tol)

        exp_var.experimental_semivariance_array = rescaled
        semivars = rescaled[:, 1]
        exp_var.experimental_semivariances = semivars
        variance = np.mean(semivars[-5:])
        exp_var.variance = variance
        exp_var.lags = self.initial_experimental_variogram.lags

        return exp_var

    def __str__(self):

        if self.was_fit:
            msg_fit = '* Model has been fitted'
        else:
            msg_fit = '* Model has not been fitted'

        if self.was_transformed:
            msg_trans = '* Model has been transformed'
        else:
            msg_trans = '* Model has not been transformed'

        msg = msg_fit + '\n' + msg_trans
        return msg
