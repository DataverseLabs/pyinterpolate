"""
Semivariogram regularization & deconvolution process.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Union, List

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.validators.common import check_limits
from pyinterpolate.semivariogram.deconvolution.aggregated_variogram import \
    regularize
from pyinterpolate.semivariogram.deconvolution.deviation import Deviation
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import \
    TheoreticalVariogram


class Deconvolution:
    """
    Class performs deconvolution of semivariogram of areal data. Whole
    procedure is based on the iterative process described in: [1].

    Steps to regularize semivariogram:

    - initialize your object (no parameters),
    - use ``fit()`` method to build the initial point support model,
    - use ``transform()`` method to perform the semivariogram regularization,
    - save deconvoluted semivariogram model with ``export_model()`` method.

    Parameters
    ----------
    verbose : bool, default = True
        Print process steps into a terminal.

    store_models : bool, default = False
        Should theoretical and regularized models be stored after each
        iteration.

    Attributes
    ----------
    ps : PointSupport
        Point support data.

    blocks : Blocks
        Blocks with aggregated data.

    deviation : Deviation
        Deviation object tracking the fit error between the initial model and
        regularized model.

    min_deviation_ratio : float
        Minimal ratio of model deviation to initial deviation when algorithm
        is stopped. Parameter must be set within the limits ``(0, 1)``. Set
        in ``transform()`` method.

    min_deviation_decrease : float
        The minimum ratio of the difference between model deviation and
        optimal deviation to the optimal deviation:
        ``|dev - opt_dev| / opt_dev``.
        Parameter must be set within the limits ``(0, 1)``. Set in
        ``transform()`` method.

    reps_deviation_decrease : int
        How many repetitions of small deviation decrease before termination
        of the algorithm.

    weights : List
        Weights applied to each lag during each regularization iteration.

    final_theoretical_model : TheoreticalVariogram
        Optimal model with the lowest deviation between this model and initial
        theoretical variogram derived from blocks.

    final_regularized_variogram : numpy array
        ``[lag, semivariance]``

    is_fit : bool, default = False
        Was model fitted? (The first step of regularization).

    is_transformed : bool, default = False
        Was model transformed?

    Methods
    -------
    fit()
        Fits areal data and the point support data into a model,
        initializes the experimental semivariogram,
        the theoretical semivariogram model,
        regularized point support model, and deviation.

    transform()
        Performs semivariogram regularization.

    fit_transform()
        Performs fit() and transform() at once.

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
    [1] Goovaerts P., Kriging and Semivariogram Deconvolution in
    the Presence of Irregular Geographical Units,
    Mathematical Geology 40(1), 101-128, 2008

    Examples
    --------
    >>> dcv = Deconvolution(verbose=True)
    >>> dcv.fit(agg_dataset=...,
    ...         point_support_dataset=...,
    ...         step_size=...,
    ...         max_range=...,
    ...         variogram_weighting_method='closest')
    >>> dcv.transform(_max_iters=5)
    >>> dcv.plot_variograms()
    >>> dcv.plot_deviations()
    >>> dcv.plot_weights()
    >>> dcv.export_model('results.csv')
    """

    def __init__(self, verbose=True, store_models=False):

        # Core data structures
        self.ps = None  # point support
        self.blocks = None  # aggregated dataset

        # Deviation and custom_weights
        self.deviation: Deviation = None
        self.min_deviation_ratio = None
        self.min_deviation_decrease = None
        self.reps_deviation_decrease = None
        self.weights = []

        # Variograms - final
        self.final_theoretical_model = None
        self.final_regularized_variogram = None

        # Control
        self.is_fit = False
        self.is_transformed = False

        self._max_iters = 0
        self._iters_deviation_decrease = 0
        self._deviation_counter = 0
        self._w_change = False
        self._verbose = verbose
        self._store_models = store_models
        self._iter = 0

        # Variograms - initial
        self._initial_regularized_variogram = None
        self._initial_theoretical_model = TheoreticalVariogram()
        self._initial_theoretical_model_prediction = None
        self._initial_experimental_variogram: ExperimentalVariogram = None

        # Variograms - optimal
        self._s2 = None  # sill of initial theoretical model squared
        self._optimal_theoretical_model: TheoreticalVariogram = None
        self._optimal_regularized_variogram = None

        # Initial variogram parameters
        self._step_size = None
        self._max_range = None
        self._nugget = None
        self._direction = None
        self._ranges = None
        self._tolerance = None
        self._weighting_method = None
        self._models_group = None

        # Debug and stability
        # List with theoretical models parameters
        self._theoretical_models = []
        # List with arrays with regularized models
        self._regularized_models = []

    def fit(self,
            blocks: Blocks,
            point_support: PointSupport,
            step_size: float,
            max_range: float,
            nugget: float = None,
            direction: float = None,
            tolerance: float = None,
            variogram_weighting_method: str = 'closest',
            models_group: Union[str, list] = 'safe') -> None:
        """
        Fits the blocks semivariogram into the point support semivariogram.
        The initial step of regularization.

        Parameters
        ----------
        blocks : Blocks
            Aggregated data | choropleth map.

        point_support : PointSupport
            Point support of ``blocks``.

        step_size : float
            Step size between lags - estimated for ``blocks``.

        max_range : float
            Maximal distance of analysis - estimated for ``blocks``.

        nugget : float, default = 0
            The nugget of a data - estimated for ``blocks``.

        direction : float (in range [0, 360]), optional
            Direction of ``blocks`` semivariogram, values from 0 to 360
            degrees:

            - 0 or 180: is E-W,
            - 90 or 270 is N-S,
            - 45 or 225 is NE-SW,
            - 135 or 315 is NW-SE.

        tolerance : float (in range [0, 1]), optional
            If ``tolerance`` is 0 then points must be placed at a single
            line with the beginning in the origin of the coordinate system
            and the direction given by y axis and direction parameter.
            If ``tolerance`` is ``> 0`` then the points are selected in
            elliptical area with major axis pointed in the same direction as
            the line for ``0`` tolerance.

            * The major axis size == ``step_size``.
            * The minor axis size is ``tolerance * step_size``
            * The baseline point is at a center of the ellipse.
            * The ``tolerance == 1`` creates an omnidirectional semivariogram.

        variogram_weighting_method : str, default = ``"closest"``
            Method used to weight error at a given lags. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger custom_weights,
            - distant: lags that are further away have bigger custom_weights,
            - dense: error is weighted by the number of point pairs within lag.

        models_group : str or list, default='safe'
            Models group to test:

            - 'all' - the same as list with all models,
            - 'safe' - ['linear', 'power', 'spherical']
            - as a list: multiple model types to test
            - as a single model type from:
                - 'circular',
                - 'cubic',
                - 'exponential',
                - 'gaussian',
                - 'linear',
                - 'power',
                - 'spherical'.
        """

        if self._verbose:
            print('Initial fit of semivariogram')

        # Update class parameters
        self.blocks = blocks
        self.ps = point_support
        self._step_size = step_size
        self._max_range = max_range

        if nugget is not None:
            self._nugget = nugget
        self._ranges = np.arange(step_size, max_range, step_size)
        if direction is not None:
            self._direction = direction
        self._tolerance = tolerance
        self._weighting_method = variogram_weighting_method
        self._models_group = models_group

        # Compute experimental variogram of areal data
        areal_centroids = self.blocks.representative_points_array()

        self._initial_experimental_variogram = ExperimentalVariogram(
            ds=areal_centroids,
            step_size=self._step_size,
            max_range=self._max_range,
            direction=self._direction,
            tolerance=self._tolerance
        )

        # Compute theoretical variogram of areal data
        self._initial_theoretical_model.autofit(
            experimental_variogram=self._initial_experimental_variogram,
            nugget=self._nugget,
            models_group=self._models_group,
            deviation_weighting=self._weighting_method
        )
        self._s2 = self._initial_theoretical_model.sill
        self._initial_theoretical_model_prediction = self._initial_theoretical_model.experimental_semivariances

        # Regularize
        self._initial_regularized_variogram = regularize(
            blocks=self.blocks,
            point_support=self.ps,
            step_size=self._step_size,
            max_range=self._max_range,
            nugget=self._nugget,
            direction=self._direction,
            tolerance=self._tolerance,
            theoretical_block_model=self._initial_theoretical_model,
            variogram_weighting_method=self._weighting_method,
            verbose=self._verbose,
            log_process=False
        )

        self.deviation = Deviation(
            self._initial_theoretical_model.yhat,
            self._initial_regularized_variogram[:, 1]
        )

        self.is_fit = True
        self._iter = 1

        if self._verbose:
            print('Regularization fit process ends')

    def transform(self,
                  max_iters=25,
                  min_deviation_ratio=0.1,
                  min_deviation_decrease=0.01,
                  reps_deviation_decrease=3):
        """
        Method performs semivariogram regularization after model fitting.

        Parameters
        ----------
        max_iters : int, default = 25
            Maximum number of iterations.

        min_deviation_ratio : float, default = 0.1
            Minimal ratio of model deviation to initial deviation when
            algorithm is stopped. Parameter must be set within the limits
            ``(0, 1)``.

        min_deviation_decrease : float, default = 0.01
            The minimum ratio of the difference between model deviation
            and optimal deviation to the optimal deviation:
            ``|dev - opt_dev| / opt_dev``.
            Parameter must be set within the limits ``(0, 1)``.

        reps_deviation_decrease : int, default = 3
            How many repetitions of small deviation decrease before
            termination of the algorithm.

        Raises
        ------
        AttributeError
            ``initial_regularized_model`` is undefined (user didn't
            perform ``fit()`` method).

        ValueError
            ``limit_deviation_ratio`` or ``minimum_deviation_decrease``
            parameters ``<= 0 or >= 1``.

        """

        print('Transform procedure starts')

        # Check if model was fitted
        self._check_fit()

        # Check limits
        check_limits(min_deviation_ratio)
        check_limits(min_deviation_decrease)

        # Update optimal models with initial models -
        # make copies to be sure that we didn't overwrite values
        self._max_iters = max_iters
        self.min_deviation_ratio = min_deviation_ratio
        self.min_deviation_decrease = min_deviation_decrease
        self.reps_deviation_decrease = reps_deviation_decrease

        initial_model_params = self._initial_theoretical_model.to_dict()
        self._optimal_theoretical_model = self._initial_theoretical_model
        self._optimal_regularized_variogram = self._initial_regularized_variogram.copy()

        # Append models if _store_models parameter is set to True
        if self._store_models:
            self._theoretical_models.append(initial_model_params)
            self._regularized_models.append(
                self._initial_regularized_variogram)

        # Start iterative procedure
        for i in trange(self._max_iters, disable=not self._verbose):
            deviation_test = self._check_deviation_gains(i)
            if deviation_test:
                if self._verbose:
                    print('Process terminated: deviation gain is too small')
                break
            else:
                # Compute new experimental values for
                # the new experimental point support model
                rescaled_experimental_variogram = self._rescale_optimal_theoretical_model()

                # Fit rescaled model to the new theoretical fn
                temp_theoretical_semivariogram_model = TheoreticalVariogram()
                temp_theoretical_semivariogram_model.autofit(
                    experimental_variogram=rescaled_experimental_variogram,
                    nugget=self._nugget,
                    models_group=self._models_group,
                    deviation_weighting=self._weighting_method
                )

                # Regularize model
                temp_regularized_variogram = regularize(
                    blocks=self.blocks,
                    step_size=self._step_size,
                    max_range=self._max_range,
                    nugget=self._nugget,
                    point_support=self.ps,
                    theoretical_block_model=temp_theoretical_semivariogram_model,
                    experimental_block_semivariances=rescaled_experimental_variogram,
                    direction=self._direction,
                    tolerance=self._tolerance,
                    variogram_weighting_method=self._weighting_method,
                    verbose=not self._verbose,
                    log_process=False
                )

                # Compute diff stats

                self.deviation.update(
                    self._initial_theoretical_model.yhat,
                    temp_regularized_variogram[:, 1]
                )

                if not self.deviation.deviation_direction():
                    self._w_change = False
                    self.deviation.set_current_as_optimal()

                    self._optimal_theoretical_model = temp_theoretical_semivariogram_model
                    self._optimal_regularized_variogram = temp_regularized_variogram

                else:
                    self._w_change = True

                self._iter = self._iter + 1

                # Append models if _store_models parameter is set to True
                if self._store_models:
                    self._theoretical_models.append(
                        temp_theoretical_semivariogram_model
                    )
                    self._regularized_models.append(
                        temp_regularized_variogram
                    )

        # Get theoretical model from regularized
        self.final_theoretical_model = self._optimal_theoretical_model
        self.final_regularized_variogram = self._optimal_regularized_variogram

        self.is_transformed = True

    def fit_transform(self,
                      blocks: Blocks,
                      point_support: PointSupport,
                      step_size: float,
                      max_range: float,
                      nugget: float = None,
                      direction: float = None,
                      tolerance: float = None,
                      variogram_weighting_method: str = 'closest',
                      models_group: Union[str, list] = 'safe',
                      max_iters=25,
                      limit_deviation_ratio=0.1,
                      minimum_deviation_decrease=0.01,
                      reps_deviation_decrease=3):
        """
        Method performs ``fit()`` and ``transform()`` operations at once.

        Fits the blocks semivariogram into the point support semivariogram.
        The initial step of regularization.

        Parameters
        ----------
        blocks : Blocks
            Aggregated data | choropleth map.

        point_support : PointSupport
            Point support of ``blocks``.

        step_size : float
            Step size between lags - estimated for ``blocks``.

        max_range : float
            Maximal distance of analysis - estimated for ``blocks``.

        nugget : float, default = 0
            The nugget of a data - estimated for ``blocks``.

        direction : float (in range [0, 360]), optional
            Direction of ``blocks`` semivariogram, values from 0 to 360
            degrees:

            - 0 or 180: is E-W,
            - 90 or 270 is N-S,
            - 45 or 225 is NE-SW,
            - 135 or 315 is NW-SE.

        tolerance : float (in range [0, 1]), optional
            If ``tolerance`` is 0 then points must be placed at a single line
            with the beginning in the origin of the coordinate system and the
            direction given by y axis and direction parameter.
            If ``tolerance`` is ``> 0`` then the bin is selected as an
            elliptical area with major axis pointed in the same direction
            as the line for ``0`` tolerance.

            * The major axis size == ``step_size``.
            * The minor axis size is ``tolerance * step_size``
            * The baseline point is at a center of the ellipse.
            * The ``tolerance == 1`` creates an omnidirectional semivariogram.

        variogram_weighting_method : str, default = ``"closest"``
            Method used to weight error at a given lags. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger custom_weights,
            - distant: lags that are further away have bigger custom_weights,
            - dense: error is weighted by the number of point pairs within lag.

        models_group : str or list, default='safe'
            Models group to test:

            - 'all' - the same as list with all models,
            - 'safe' - ['linear', 'power', 'spherical']
            - as a list: multiple model types to test
            - as a single model type from:
                - 'circular',
                - 'cubic',
                - 'exponential',
                - 'gaussian',
                - 'linear',
                - 'power',
                - 'spherical'.

        max_iters : int, default = 25
            Maximum number of iterations.

        limit_deviation_ratio : float, default = 0.01
            Minimal ratio of model deviation to initial deviation when
            algorithm is stopped. Parameter must be set
            within the limits ``(0, 1)``.

        minimum_deviation_decrease : float, default = 0.001
            The minimum ratio of the difference between model deviation
            and optimal deviation to the optimal deviation:
            ``|dev - opt_dev| / opt_dev``. The parameter must be set within
            the limits ``(0, 1)``.

        reps_deviation_decrease : int, default = 3
            How many repetitions of small deviation decrease before
            termination of the algorithm.
        """

        self.fit(blocks=blocks,
                 point_support=point_support,
                 step_size=step_size,
                 max_range=max_range,
                 nugget=nugget,
                 direction=direction,
                 tolerance=tolerance,
                 variogram_weighting_method=variogram_weighting_method,
                 models_group=models_group)

        self.transform(max_iters=max_iters,
                       min_deviation_ratio=limit_deviation_ratio,
                       min_deviation_decrease=minimum_deviation_decrease,
                       reps_deviation_decrease=reps_deviation_decrease)

    def export_model_to_json(self, fname: str):
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
            raise RuntimeError('You cannot export any '
                               'model if you not transform data.')

        self.final_theoretical_model.to_json(fname)

    def plot_variograms(self):
        """
        Function shows experimental semivariogram, theoretical
        semivariogram and regularized semivariogram after
        semivariogram regularization (transforming).
        """
        lags = self._initial_experimental_variogram.lags

        plt.figure(figsize=(12, 6))
        plt.plot(lags,
                 self._initial_experimental_variogram.semivariances, 'bo')
        plt.plot(lags,
                 self._initial_theoretical_model.predict(lags), color='r',
                 linestyle='--')

        if self.final_regularized_variogram is not None:
            plt.plot(lags, self.final_regularized_variogram[:, 1], 'go')

            plt.plot(lags,
                     self.final_theoretical_model.predict(lags),
                     color='black',
                     linestyle='dotted')
            plt.legend(['Experimental semivariogram of areal data',
                        'Initial Semivariogram of areal data',
                        'Regularized data points, iteration {}'.format(
                            self._iter),
                        'Optimized theoretical point support model'])
            plt.title(
                'Semivariograms comparison. Deviation value: {}'.format(
                    self.deviation.optimal_deviation
                )
            )
        else:
            plt.legend(['Experimental semivariogram of areal data',
                        'Initial Semivariogram of areal data'])
            plt.title('Semivariograms comparison')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    def plot_deviation_change(self, normalized=True):
        self.deviation.plot(normalized=normalized)

    def plot_weights_change(self, averaged=True):

        plt.figure(figsize=(12, 6))
        iter_no = np.arange(len(self.weights))
        if averaged:
            plt.plot(
                iter_no,
                [np.mean(weight) for weight in self.weights]
            )
            plt.title('Average custom_weights change')
            plt.ylabel('Average weight')
        else:
            legend = []
            for idx, lag in enumerate(self._optimal_theoretical_model.lags):
                legend.append(f'Lag: {lag:.4f}')
                plt.plot(
                    iter_no,
                    [weight[idx] for weight in self.weights],
                    'o:'
                )

            plt.title('Lag custom_weights change')
            plt.ylabel('Weight')
            plt.legend(legend)

        plt.xlabel('Iteration')

        plt.show()

    def _check_fit(self):
        if self._initial_regularized_variogram is None:
            msg = ('The initial regularized model is undefined. Perform fit()'
                   ' before transformation!')
            raise AttributeError(msg)

    def _check_deviation_gains(self, iter_no: int):
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
            dev_decrease = self.deviation.calculate_deviation_decrease()

            # TODO this condition is very hard to met, have to be changed
            if dev_decrease < 0 and abs(
                    dev_decrease) <= self.min_deviation_decrease:
                self._deviation_counter = self._deviation_counter + 1
                if self._deviation_counter == self._iters_deviation_decrease:
                    return True
                return False
            else:
                self._deviation_counter = 0
                return False

    def _deviation_ratio(self) -> bool:
        """
        The model deviation to initial deviation.

        Returns
        -------
        : bool
        """
        dev_ratio = self.deviation.calculate_deviation_ratio()
        if dev_ratio <= self.min_deviation_ratio:
            return True
        return False

    def _rescale_optimal_theoretical_model(self) -> np.ndarray:
        r"""
        Function rescales points derived from the optimal theoretical model
        and creates new experimental values based on the equation:

        $$\gamma_{res}(h) = \gamma_{opt}(h) \times w(h)$$

        $$w(h) =
          1 + \frac{\gamma_{v}^{exp}(h)-\gamma_{v}^{opt}}{s^{2}\sqrt{_iter}}$$

        where:

        - $\gamma_{v}^{exp}(h)$ : theoretical model fitted to blocks
            (1st _iter), then point support model that has been derived from
            a rescaled values.
        - $\gamma_{v}^{opt}$ : optimal point support model (after
            regularization),
        - $w(h)$ : custom_weights vector (each record is a weight applied
            to a specific lag),
        - $s$ - sill of the theoretical model fitted to the blocks,
        - $_iter$ - iteration number.

        Returns
        -------
        rescaled : numpy array
                    Rescalled point support model.

        """

        y_opt_h = self._optimal_theoretical_model.predict(self._ranges)

        if not self._w_change:
            denom = self._s2 * np.sqrt(self._iter)
            numer = (self._initial_theoretical_model_prediction -
                     self._optimal_regularized_variogram[:, 1])
            w = 1 + (numer / denom)
        else:
            w = 1 + (self.weights[-1] - 1) / 2

        rescaled = np.zeros_like(self._optimal_regularized_variogram)
        rescaled[:, 0] = self._optimal_regularized_variogram[:, 0]
        rescaled[:, 1] = y_opt_h * w

        self.weights.append(w)

        return rescaled

    def _rescaled_to_exp_variogram(
            self,
            rescaled: np.ndarray
    ) -> ExperimentalVariogram:
        """
        Function sets new experimental variogram based on the rescaled
        semivariances; variance of experimental variogram is calculated
        as the mean of the last 5 rescaled semivariances.

        Parameters
        ----------
        rescaled : numpy array
            Rescaled semivariances.

        Returns
        -------
        exp_var : ExperimentalVariogram
            Experimental variogram of rescaled semivariances.
        """
        exp_var = ExperimentalVariogram(
            ds=self._initial_experimental_variogram.ds,
            custom_bins=self._initial_experimental_variogram.lags,
            custom_weights=self._initial_experimental_variogram.custom_weights,
            direction=self._initial_experimental_variogram.direction,
            tolerance=self._initial_experimental_variogram.tolerance)

        exp_var.semivariances = rescaled
        variance = np.mean(exp_var.semivariances[:, 1][-5:])
        exp_var.variance = variance

        return exp_var

    def __str__(self):
        """
        Modeling status.

        Returns
        -------
        : str
        """
        if self.is_fit:
            msg_fit = '* Model has been fitted'
        else:
            msg_fit = '* Model has not been fitted'

        if self.is_transformed:
            msg_trans = '* Model has been transformed'
        else:
            msg_trans = '* Model has not been transformed'

        msg = msg_fit + '\n' + msg_trans
        return msg
