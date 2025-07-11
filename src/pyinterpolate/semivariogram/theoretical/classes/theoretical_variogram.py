import json
from typing import Union, Tuple, Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from prettytable import PrettyTable

from pyinterpolate.core.data_models.theoretical_variogram import \
    TheoreticalVariogramModel, SemivariogramErrorsModel
from pyinterpolate.evaluate.metrics import forecast_bias, \
    weighted_root_mean_squared_error, root_mean_squared_error, \
    mean_absolute_error, symmetric_mean_absolute_percentage_error
from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.semivariogram.theoretical.spatial_dependency_index import calculate_spatial_dependence_index
from pyinterpolate.semivariogram.theoretical.variogram_models import (
    ALL_MODELS, SAFE_MODELS, TheoreticalModelFunction
)
from pyinterpolate.transform.select_points import create_min_max_array


class TheoreticalVariogram:
    """Theoretical model of a spatial dissimilarity.

    Parameters
    ----------
    model_params : Union[dict, TheoreticalVariogramModel], optional
        Dictionary with
        ``'nugget'``, ``'sill'``, ``'rang'`` and ``'variogram_model_type'``.

    protect_from_overwriting : bool, default = True
        Protect model parameters from overwriting.

    verbose : bool, default = False
        Show `autofit()` iteration results.

    Attributes
    ----------
    experimental_variogram : ExperimentalVariogram, optional
        Empirical Variogram class and its attributes.

    experimental_semivariances : numpy array, optional
        Experimental semivariances.

    yhat : numpy array, optional
        Predicted semivariances.

    model_type : str, optional
        The name of the chosen model.

    nugget : float, default=0
        The nugget parameter (bias at the zero distance).

    sill : float, default=0
        A value at which dissimilarity is close to its maximum if model is
        bounded. Otherwise, it is usually close to the observations variance.

    rang : float, default=0
        The semivariogram range is a distance at which spatial correlation
        might be observed. It shouldn't be set at a distance larger than
        a half of a study extent.

    direction : float, in range [0, 360], optional
        The direction of a semivariogram.  If not given then semivariogram
        is isotropic.

    rmse : float, default=0
        Root mean squared error of the difference between the empirical
        observations and the modeled curve.

    mae : bool, default=True
        Mean Absolute Error of a model.

    bias : float, default=0
        Forecast Bias of the modeled semivariogram vs experimental points.
        Large positive value means that the estimated model underestimates
        predictions. A large negative value means that model overestimates
        predictions.

    smape : float, default=0
        Symmetric Mean Absolute Percentage Error of the prediction -
        values from 0 to 100%.

    spatial_dependency_ratio : float, optional
        The ratio of nugget vs sill multiplied by 100. Levels from 0 to 25
        indicate strong spatial dependency, from 25 to 75  - moderate spatial
        dependency, from 75 to 95 - weak spatial dependency, and above
        the process is considered to be not spatially-depended.

    spatial_dependency_strength : str, default = "Unknown"
        Descriptive indicator of spatial dependency strength based on the
        ``spatial_dependency_level``. It could be:

        * ``unknown`` if ratio is ``None``,
        * ``strong`` if ratio is below 25,
        * ``moderate`` if ratio is between 25 and 75,
        * ``weak`` if ratio is between 75 and 95,
        * ``no spatial dependency`` if ratio is greater than 95.

    protect_from_overwriting : bool, default = True
        Protect model parameters from overwriting.

    verbose : bool, default = False
        Show `autofit()` results.

    Methods
    -------
    fit()
        Fits experimental variogram data into theoretical model.

    autofit()
        The same as fit but tests multiple nuggets, ranges, sills and models.

    calculate_model_error()
        Evaluates the model performance against experimental values.

    to_dict()
        Store model parameters in a dict.

    from_dict()
        Read model parameters from a dict.

    to_json()
        Save model parameteres in a JSON file.

    from_json()
        Read model parameters from a JSON file.

    plot()
        Shows theoretical model.

    See Also
    --------
    ExperimentalVariogram : class to calculate experimental variogram and more.

    Examples
    --------
    TODO
    """

    def __init__(self,
                 model_params: Union[dict, None] = None,
                 protect_from_overwriting=True,
                 verbose=False):

        # Model
        self._study_max_range = None

        # Model parameters
        self.lags = None
        self.experimental_variogram = None

        self.experimental_semivariances = None
        self.yhat = None

        self.model_type = None

        self.nugget = 0.
        self.rang = 0.
        self.sill = 0.

        self.direction = None

        if model_params is not None:
            self._set_model_parameters(model_params)

        # Error
        self.deviation_weighting = None
        self.rmse = 0.
        self.bias = 0.
        self.smape = 0.
        self.mae = 0.

        # Spatial dependency
        self.spatial_dependency_ratio = None
        self.spatial_dependency_strength = 'Unknown'

        # Class control
        self.protect_from_overwriting = protect_from_overwriting
        self.verbose = verbose

        # Autofit parameters
        self._models_group = {
            'safe': SAFE_MODELS,
            'all': ALL_MODELS
        }

    @property
    def name(self):
        """
        Returns theoretical model name.
        """
        return self.model_type

    # Core functions

    def fit(self,
            experimental_variogram: Union[ExperimentalVariogram, np.ndarray],
            model_type: str,
            sill: float,
            rang: float,
            nugget=0.,
            direction=None) -> Tuple[np.ndarray, dict]:
        """
        Fits theoretical model into experimental semivariances.

        Parameters
        ----------
        experimental_variogram : ExperimentalVariogram
            Experimental variogram model or array with lags and semivariances.

        model_type : str
            The name of the model to check. Available models:

            - 'circular',
            - 'cubic',
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical'.

        sill : float, default=0
            A value at which dissimilarity is close to its maximum if model is
            bounded. Otherwise, it is usually close to observations variance.

        rang : float, default=0
            The semivariogram range is a distance at which spatial correlation
            exists. It shouldn't be set at a distance larger than a half
            of a study extent.

        nugget : float, default=0.
            Nugget parameter (bias at a zero distance).

        direction : float, in range [0, 360], default=None
            The direction of a semivariogram. If ``None`` given then
            semivariogram is isotropic. This parameter is required if
            passed experimental variogram is stored as a numpy array.

        Raises
        ------
        AttributeError : Semivariogram parameters could be overwritten

        Returns
        -------
        theoretical_values, error: Tuple[ numpy array, dict ]
            ``[ theoretical semivariances, {'rmse bias smape mae'}]``

        """
        self.__are_parameters_fit()

        # Update variogram attributes
        self.__update_experimental_variogram_attributes(
            experimental_variogram=experimental_variogram,
            direction=direction
        )

        theoretical_values = self._fit_model(model_type, nugget, sill, rang)

        # Estimate errors
        _error = self.calculate_model_error(theoretical_values,
                                            rmse=True,
                                            bias=True,
                                            smape=True)

        self.__update_class_attrs(
            yhat=theoretical_values,
            model_type=model_type,
            nugget=nugget,
            sill=sill,
            rang=rang,
            **_error
        )

        # Update spatial dependency
        self.__update_spatial_dependency_index()

        return theoretical_values, _error

    def autofit(self,
                experimental_variogram: Union[
                    ExperimentalVariogram, np.ndarray
                ],
                models_group: Union[str, list] = 'safe',
                nugget=None,
                min_nugget=0,
                max_nugget=0.5,
                number_of_nuggets=16,
                rang=None,
                min_range=0.1,
                max_range=0.5,
                number_of_ranges=16,
                sill=None,
                n_sill_values=5,
                sill_from_variance=False,
                min_sill=0.5,
                max_sill=2,
                number_of_sills=16,
                direction=None,
                error_estimator='rmse',
                deviation_weighting='equal',
                return_params=True) -> Optional[TheoreticalVariogramModel]:
        """
        Method finds the optimal range, sill and model (function)
        of theoretical semivariogram.

        Parameters
        ----------
        experimental_variogram : ExperimentalVariogram
            Experimental variogram model or array with lags and semivariances.

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

        nugget : float, optional
            Nugget (bias) of a variogram. If given then it is
            fixed to this value.

        min_nugget : float, default = 0
            The minimum nugget as the ratio of the parameter to
            the first lag variance.

        max_nugget : float, default = 0.5
            The maximum nugget as the ratio of the parameter to
            the first lag variance.

        number_of_nuggets : int, default = 16
            How many equally spaced nuggets tested between
            ``min_nugget`` and ``max_nugget``.

        rang : float, optional
            If given, then range is fixed to this value.

        min_range : float, default = 0.1
            The minimal fraction of a variogram range,
            ``0 < min_range <= max_range``.

        max_range : float, default = 0.5
            The maximum fraction of a variogram range,
            ``min_range <= max_range <= 1``. Parameter ``max_range`` greater
            than **0.5** raises warning.

        number_of_ranges : int, default = 16
            How many equally spaced ranges are tested between
            ``min_range`` and ``max_range``.

        sill : float, default = None
            If given, then sill is fixed to this value.

        n_sill_values : int, default = 5
            The last n experimental semivariance records for sill estimation.
            (Used only when ``sill_from_variance`` is set to ``False``).

        sill_from_variance : bool, default = False
            Estimate sill from the variance (semivariance at distance 0).

        min_sill : float, default = 1
            The minimal fraction of the value chosen with the sill estimation
            method. The value is: for ``sill_from_values`` - the mean of
            the last ``n_sill_values`` number of experimental semivariances,
            for ``sill_from_variance`` - the experimental variogram variance.

        max_sill : float, default = 5
            The maximum fraction of the value chosen with the sill estimation
            method. The value is: for ``sill_from_values`` - the mean of
            the last ``n_sill_values`` number of experimental semivariances,
            for ``sill_from_variance`` - the experimental variogram variance.

        number_of_sills : int, default = 16
            How many equally spaced sill values are tested between
            ``min_sill`` and ``max_sill``.

        direction : float, in range [0, 360], default=None
            The direction of a semivariogram. If ``None`` given then
            semivariogram is isotropic. This parameter is required if passed
            experimental variogram is stored as a numpy array.

        error_estimator : str, default = 'rmse'
            A model error estimation method. Available options are:

            - 'rmse': Root Mean Squared Error,
            - 'mae': Mean Absolute Error,
            - 'bias': Forecast Bias,
            - 'smape': Symmetric Mean Absolute Percentage Error.

        deviation_weighting : str, default = "equal"
            The name of the method used to weight error at a given lags. Works
            only with RMSE. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger weights,
            - distant: lags that are further away have bigger weights,
            - dense: error is weighted by the number of point pairs within lag.

        return_params : bool, default = True
            Returns model.

        Returns
        -------
        theoretical_variogram_model : TheoreticalVariogramModel
            See ``TheoreticalVariogramModel`` class.


        Raises
        ------
        AttributeError
            Method is invoked on the calculated variogram.

        ValueError
            Raised when wrong nugget, range, or sill limits are passed.

        KeyError
            Raised when wrong error type is provided by the users.
        """
        self.__are_parameters_fit()

        self.deviation_weighting = deviation_weighting

        # Update variogram attributes
        self.__update_experimental_variogram_attributes(
            experimental_variogram=experimental_variogram,
            direction=direction
        )

        # Models for test
        if isinstance(models_group, str):
            models = self._models_group.get(models_group, False)
            if not models:
                models = [models_group]
        else:
            # User has provided list with model types
            models = models_group

        # Set nuggets, ranges and sills
        nugget_ranges = self._prepare_nugget_ranges(nugget,
                                                    number_of_nuggets,
                                                    min_nugget,
                                                    max_nugget)

        distance_ranges = self._prepare_distance_ranges(
            rang,
            number_of_ranges,
            min_range,
            max_range
        )

        sill_ranges = self._prepare_sill_ranges(
            sill=sill,
            n_sill_values=n_sill_values,
            sill_from_variance=sill_from_variance,
            number_of_sills=number_of_sills,
            min_sill=min_sill,
            max_sill=max_sill
        )

        # Get errors
        _errors_keys = self._get_err_keys(error_estimator)

        # Initlize parameters
        theoretical_variogram_model = self._autofit_grid_search(
            models=models,
            nugget_ranges=nugget_ranges,
            distance_ranges=distance_ranges,
            sill_ranges=sill_ranges,
            errors_keys=_errors_keys,
            error_estimator=error_estimator,
            deviation_weighting=deviation_weighting
        )

        self.__update_class_attrs(
            yhat=theoretical_variogram_model.yhat,
            model_type=theoretical_variogram_model.variogram_model_type,
            nugget=theoretical_variogram_model.nugget,
            sill=theoretical_variogram_model.sill,
            rang=theoretical_variogram_model.rang,
            **theoretical_variogram_model.errors.model_dump()
        )

        # Update spatial dependency
        self.__update_spatial_dependency_index()

        if return_params:
            return theoretical_variogram_model

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """
        Method predicts semivariances from distances using fitted
        semivariogram model.

        Parameters
        ----------
        distances : numpy array
            Distances between points.

        Returns
        -------
        predicted : numpy array
            Predicted semivariances.
        """

        model = TheoreticalModelFunction(
            lags=distances,
            nugget=self.nugget,
            sill=self.sill,
            rang=self.rang
        )

        predicted = model.fit_predict(
            model_type=self.model_type
        )

        return predicted

    # Plotting and visualization
    def plot(self, experimental=True):
        """
        Method plots theoretical semivariogram.

        Parameters
        ----------
        experimental : bool
            Plots experimental observations with theoretical semivariogram.

        Raises
        ------
        AttributeError
            Model is not fitted yet, nothing to plot.
        """
        if self.yhat is None:
            if self._params_are_given():
                self._plot_from_params()
            else:
                raise AttributeError('Model has not been trained, '
                                     'nothing to plot.')
        else:
            legend = ['Theoretical Model']
            plt.figure(figsize=(12, 6))

            if experimental:
                plt.scatter(self.lags,
                            self.experimental_semivariances,
                            marker='8', c='#66c2a5')
                legend = ['Experimental Semivariances', 'Theoretical Model']

            plt.plot(self.lags, self.yhat, '--', color='#fc8d62')
            plt.legend(legend)
            plt.xlabel('Distance')
            plt.ylabel('Variance')
            plt.show()

    # Evaluation
    def calculate_model_error(self,
                              fitted_values: np.ndarray,
                              rmse=True,
                              bias=True,
                              mae=True,
                              smape=True,
                              deviation_weighting='equal') -> dict:
        """
        Method calculates error associated with a difference between
        the theoretical model and the experimental semivariances.

        Parameters
        ----------
        fitted_values : numpy array

        rmse : bool, default=True
            Root Mean Squared Error of a model.

        bias : bool, default=True
            Forecast Bias of a model.

        mae : bool, default=True
            Mean Absolute Error of a model.

        smape : bool, default=True
            Symmetric Mean Absolute Percentage Error of a model.

        deviation_weighting : str, default = "equal"
            The name of the method used to **weight errors at a given lags**.
            Works only with RMSE. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger weights,
            - distant: lags that are further away have bigger weights,
            - dense: error is weighted by the number of point pairs within
              a lag.

        Returns
        -------
        model_errors : Dict
            Computed errors: rmse, bias, mae, smape.

        Raises
        ------
        MetricsTypeSelectionError
            User has set all error types to ``False``.
        """

        model_error = {
            'rmse': np.nan,
            'bias': np.nan,
            'mae': np.nan,
            'smape': np.nan
        }

        _real_values = self.experimental_semivariances.copy()

        # Get Forecast Biast
        if bias:
            _fb = forecast_bias(fitted_values, _real_values)
            model_error['bias'] = _fb

        # Get RMSE
        if rmse:
            if deviation_weighting != 'equal':
                if deviation_weighting == 'dense':
                    ppl = self.experimental_variogram.points_per_lag
                    _rmse = weighted_root_mean_squared_error(
                        fitted_values,
                        _real_values,
                        deviation_weighting,
                        lag_points_distribution=ppl)
                else:
                    _rmse = weighted_root_mean_squared_error(
                        fitted_values,
                        _real_values,
                        deviation_weighting)
            else:
                _rmse = root_mean_squared_error(
                    fitted_values,
                    _real_values)
            model_error['rmse'] = _rmse

        # Get MAE
        if mae:
            _mae = mean_absolute_error(fitted_values, _real_values)
            model_error['mae'] = _mae

        # Get SMAPE
        if smape:
            _smape = symmetric_mean_absolute_percentage_error(fitted_values,
                                                              _real_values)
            model_error['smape'] = _smape

        return model_error

    # I/O
    def to_dict(self) -> dict:
        """Method exports the theoretical variogram parameters to dictionary.

        Returns
        -------
        model_parameters : Dict
            Dictionary with model's ``'variogram_model_type'``, ``'nugget'``,
            ``'sill'``, ``'rang'`` and ``'direction'``.

        Raises
        ------
        AttributeError
            The model parameters have not been derived yet.
        """

        if self.yhat is None:
            if self.model_type is None:
                msg = ('Model is not set yet, '
                       'cannot export model parameters to dict')
                raise AttributeError(msg)

        model_params = TheoreticalVariogramModel(
            nugget=self.nugget,
            sill=self.sill,
            rang=self.rang,
            variogram_model_type=self.model_type,
            direction=self.direction
        )

        return model_params.model_dump()

    def from_dict(self, parameters: dict):
        """Method updates model with a given set of parameters.

        Parameters
        ----------
        parameters : Dict
            Dictionary with model's: ``'variogram_model_type', 'nugget',
            'sill', 'range', 'direction'``.
        """

        self._set_model_parameters(parameters)

    def to_json(self, fname: str):
        """
        Method stores semivariogram parameters into a JSON file.

        Parameters
        ----------
        fname : str
            JSON file name.
        """

        json_output = self.to_dict()

        with open(fname, 'w') as fout:
            json.dump(json_output, fout)

    def from_json(self, fname: str):
        """
        Method reads data from a JSON file.

        Parameters
        ----------
        fname : str
            JSON file name.
        """

        with open(fname, 'r') as fin:
            json_input = json.load(fin)

        self._set_model_parameters(json_input)

    def __printed_results(self):
        is_model_none = self.model_type is None
        is_rang_0 = self.rang == 0
        is_sill_0 = self.sill == 0

        check_validity = is_model_none and is_sill_0 and is_rang_0

        if check_validity:
            return ('Theoretical model is not calculated yet. '
                    'Use fit() or autofit() methods to build or find a model '
                    'or import model with from_dict() or from_json() methods.')
        else:
            title = ('* Selected model: ' +
                     f'{self.model_type}'.capitalize() +
                     ' model')
            msg_nugget = f'* Nugget: {self.nugget}'
            msg_sill = f'* Sill: {self.sill}'
            msg_range = f'* Range: {self.rang}'
            msg_spatial_dependency = (f'* Spatial Dependency Strength is '
                                      f'{self.spatial_dependency_strength}')
            mean_bias_msg = f'* Mean Bias: {self.bias}'
            mean_rmse_msg = f'* Mean RMSE: {self.rmse}'
            error_weighting = (f'* Error-lag weighting method: '
                               f'{self.deviation_weighting}')

            text_list = [title,
                         msg_nugget,
                         msg_sill,
                         msg_range,
                         msg_spatial_dependency,
                         mean_bias_msg,
                         mean_rmse_msg,
                         error_weighting]

            header = '\n'.join(text_list) + '\n\n'

            if self.experimental_semivariances is not None:
                # Build pretty table
                pretty_table = PrettyTable()
                pretty_table.field_names = ["lag",
                                            "theoretical",
                                            "experimental",
                                            "bias (real-yhat)"]

                records = []
                for idx, record in enumerate(self.experimental_semivariances):
                    lag = self.lags[idx]
                    experimental_semivar = record
                    theoretical_semivar = self.yhat[idx]
                    bias = experimental_semivar - theoretical_semivar
                    records.append([lag,
                                    theoretical_semivar,
                                    experimental_semivar,
                                    bias])

                pretty_table.add_rows(records)

                msg = header + '\n' + pretty_table.get_string()
                return msg
            else:
                return header

    def __repr__(self):
        return self.__printed_results()

    def __str__(self):
        return self.__printed_results()

    def _autofit_grid_search(
            self,
            models: ArrayLike,
            nugget_ranges: ArrayLike,
            distance_ranges: ArrayLike,
            sill_ranges: ArrayLike,
            errors_keys: Dict,
            error_estimator: str,
            deviation_weighting: str
    ) -> TheoreticalVariogramModel:
        """
        Theoretical model grid search

        Parameters
        ----------
        models : ArrayLike
            Theoretical model to test.

        nugget_ranges : ArrayLike
            Nuggets to test.

        distance_ranges : ArrayLike
            Variogram ranges to test.

        sill_ranges : ArrayLike
            Sills to test.

        errors_keys : Dict
            Deviation parameters to test (rmse, bias, smape, mae).

        error_estimator : str
            Deviation parameter used to select the best model.

        deviation_weighting : str
            Method used to weight error at a given lags.

        Returns
        -------
        : TheoreticalVariogramModel
            The optimal semivariogram model, and the errors of fit.
        """
        # Initialize error
        err_val = np.inf

        # Initlize parameters
        theoretical_variogram_model = TheoreticalVariogramModel(
            nugget=0,
            sill=0,
            rang=0,
            variogram_model_type='None'
        )

        parameters_space = self.__get_parameters_space(
            models=models,
            nuggets=nugget_ranges,
            ranges=distance_ranges,
            sills=sill_ranges
        )

        for rec in parameters_space:
            _mtype = rec[0]
            _nugg = rec[1]
            _rang = rec[2]
            _sill = rec[3]

            # Create model
            _fitted_model = self._fit_model(
                model_type=_mtype,
                nugget=_nugg,
                sill=_sill,
                rang=_rang
            )

            # Calculate Error
            _err = self.calculate_model_error(
                _fitted_model,
                **errors_keys,
                deviation_weighting=deviation_weighting
            )

            if self.verbose:
                self.__print_autofit_info(_mtype,
                                          _nugg,
                                          _sill,
                                          _rang,
                                          error_estimator,
                                          _err[error_estimator])

            # Check if model is better than the previous
            if _err[error_estimator] < err_val:
                err_val = _err[error_estimator]

                theoretical_variogram_model.variogram_model_type = _mtype
                theoretical_variogram_model.nugget = _nugg
                theoretical_variogram_model.sill = _sill
                theoretical_variogram_model.rang = _rang
                theoretical_variogram_model.yhat = _fitted_model
                theoretical_variogram_model.errors = SemivariogramErrorsModel(
                    **{error_estimator: err_val}
                )

        return theoretical_variogram_model

    def _fit_model(self,
                   model_type: str,
                   nugget: float,
                   sill: float,
                   rang: float) -> np.ndarray:
        """Method fits selected model.

        Parameters
        ----------
        model_type : str
            The name of a model.

        nugget : float

        sill : float

        rang : float

        Returns
        -------
        : numpy array
            Predicted semivariances.
        """

        _input = {
            'lags': self.lags,
            'nugget': nugget,
            'sill': sill,
            'rang': rang
        }

        theoretical_model = TheoreticalModelFunction(**_input)
        fitted = theoretical_model.fit_predict(
            model_type=model_type
        )
        return fitted

    def _prepare_distance_ranges(self,
                                 rang: float = None,
                                 number_of_ranges: int = None,
                                 min_range: float = None,
                                 max_range: float = None) -> ArrayLike:
        """
        Method prepares distance ranges for the model.

        Parameters
        ----------
        rang : float, optional
            Baseline range.

        number_of_ranges : int, optional
            Number of possible ranges to test.

        min_range : float, optional
            Minimum range.

        max_range : float, optional
            Maximum range.

        Returns
        -------
        : numpy array
            Ranges.
        """
        if rang is None:
            self.__validate_distance_ranges(min_range, max_range)

            if self._study_max_range is None:
                if isinstance(self.experimental_variogram,
                              ExperimentalVariogram):
                    coordinates = self.experimental_variogram.ds[:, :-1]
                    self._study_max_range = self._get_study_range(
                        input_coordinates=coordinates
                    )
                else:
                    self._study_max_range = self.lags[-1]
            min_max_ranges = create_min_max_array(
                self._study_max_range, min_range, max_range, number_of_ranges
            )
        else:
            min_max_ranges = [rang]

        return min_max_ranges

    def _prepare_nugget_ranges(self,
                               nugget: float = None,
                               number_of_nuggets: int = None,
                               min_nugget: float = None,
                               max_nugget: float = None) -> ArrayLike:
        if nugget is None:
            self.__validate_nugget_ranges(min_nugget, max_nugget)
            nugget_rng_min = self.experimental_semivariances[0] * min_nugget
            nugget_rng_max = self.experimental_semivariances[0] * max_nugget
            nugget_ranges = np.linspace(nugget_rng_min,
                                        nugget_rng_max,
                                        number_of_nuggets)
        else:
            nugget_ranges = [nugget]
        return nugget_ranges

    def _prepare_sill_ranges(self,
                             sill: float = None,
                             n_sill_values: int = 5,
                             sill_from_variance: bool = False,
                             number_of_sills: int = None,
                             min_sill: float = None,
                             max_sill: float = None):
        """
        Method prepares sill ranges for the model.

        Parameters
        ----------
        sill : float, optional
            Baseline sill.

        n_sill_values : int, default=5
            Number of the last N experimental semivariances to use for sill
            estimation.

        sill_from_variance : bool, default = False
            Should set sill to the variance of a dataset?

        number_of_sills : int, optional
            Number of possible sills to test.

        min_sill : float, optional
            Minimum sill.

        max_sill : float, optional
            Maximum sill.

        Returns
        -------
        : numpy array
            Sills.
        """
        if sill is None:
            self.__validate_sill_ranges(min_sill, max_sill)
            if sill_from_variance:
                var_sill = self.experimental_variogram.variance
            else:
                var_sill = np.mean(
                    self.experimental_semivariances[-n_sill_values:]
                )

            min_max_sill = create_min_max_array(var_sill,
                                                min_sill,
                                                max_sill,
                                                number_of_sills)
        else:
            min_max_sill = [sill]

        return min_max_sill

    def _set_model_parameters(
            self,
            model_params: Union[dict, TheoreticalVariogramModel]
    ):
        """
        Sets model parameters.

        Parameters
        ----------
        model_params : Union[dict, TheoreticalVariogramModel]
            Parameters of the fitted semivariogram.
        """
        if isinstance(model_params, dict):
            model_params = TheoreticalVariogramModel(
                **model_params
            )

        # Update parameters
        self.model_type = model_params.variogram_model_type
        self.nugget = model_params.nugget
        self.sill = model_params.sill
        self.rang = model_params.rang
        self.direction = model_params.direction

        self.__update_spatial_dependency_index()

    def __are_parameters_fit(self):
        if self.sill > 0 and self.rang > 0:
            if self.protect_from_overwriting:
                msg = ('Semivariogram parameters have been set, '
                       'you are going to overwrite them. If you want to '
                       'overwrite semivariogram parameters then set '
                       '"protect_from_overwriting" parameter to False '
                       'during class initialization.')
                raise AttributeError(msg)

    def __update_class_attrs(self,
                             yhat: np.ndarray,
                             model_type: str,
                             nugget: float,
                             sill: float,
                             rang: float,
                             rmse: float = None,
                             bias: float = None,
                             mae: float = None,
                             smape: float = None):
        """Method updates class attributes"""
        self.yhat = yhat
        self.model_type = model_type
        self.nugget = nugget
        self.sill = sill
        self.rang = rang
        self.rmse = rmse
        self.mae = mae
        self.smape = smape
        self.bias = bias

    def __update_experimental_variogram_attributes(self,
                                                   experimental_variogram,
                                                   direction):
        """Updates experimental variogram, lags, and direction"""

        if isinstance(experimental_variogram, ExperimentalVariogram):
            self.experimental_variogram = experimental_variogram
            self.lags = experimental_variogram.lags
            self.experimental_semivariances = experimental_variogram.semivariances
            self.direction = experimental_variogram.direction
        elif isinstance(experimental_variogram, np.ndarray):
            self.experimental_semivariances = experimental_variogram[:, 1]
            self.lags = experimental_variogram[:, 0]
            self.direction = direction
        else:
            raise TypeError('Unexpected Experimental Variogram data type')

    def __update_spatial_dependency_index(self):
        """
        Method updates spatial dependency of a fitted variogram.
        """
        if self.nugget > 0:
            index_ratio, index_strength = calculate_spatial_dependence_index(
                self.nugget, self.sill
            )
        else:
            index_ratio = np.inf
            index_strength = 'Undefined: nugget equal to 0, cannot estimate'
        self.spatial_dependency_ratio = index_ratio
        self.spatial_dependency_strength = index_strength

    @staticmethod
    def _get_err_keys(err_name: str) -> Dict:
        err_dict = {
            'rmse': False,
            'bias': False,
            'smape': False,
            'mae': False
        }

        if err_name in err_dict.keys():
            err_dict[err_name] = True
            return err_dict
        else:
            msg = (f'Defined error {err_name} not exists. '
                   f'Use one of {list(err_dict.keys())} instead.')
            raise KeyError(msg)

    @staticmethod
    def _get_study_range(input_coordinates: np.ndarray) -> float:
        """Function calculates max possible range of a study area.

        Parameters
        ----------
        input_coordinates : numpy array
            [y, x] or [rows, cols]

        Returns
        -------
        study_range : float
            The extent of a study area.
        """

        min_x = min(input_coordinates[:, 1])
        max_x = max(input_coordinates[:, 1])
        min_y = min(input_coordinates[:, 0])
        max_y = max(input_coordinates[:, 0])

        study_range = (max_x - min_x) ** 2 + (max_y - min_y) ** 2
        study_range = np.sqrt(study_range)
        return study_range

    @staticmethod
    def __get_parameters_space(models, nuggets, ranges, sills) -> List:
        parameters = []
        for m in models:
            for n in nuggets:
                for r in ranges:
                    for s in sills:
                        p = (m, n, r, s)
                        parameters.append(p)
        return parameters

    @staticmethod
    def __print_autofit_info(model_name: str,
                             nugget: float,
                             sill: float,
                             rang: float,
                             err_type: str,
                             err_value: float):
        msg_core = (f'Model {model_name},\n'
                    f'Model Parameters - nugget: {nugget:.2f},\n'
                    f'sill: {sill:.4f},\n'
                    f'range: {rang:.4f},\n'
                    f'Model Error {err_type}: {err_value}\n')
        print(msg_core)

    @staticmethod
    def __validate_distance_ranges(min_dist_range: float,
                                   max_dist_range: float):
        # Check if min is lower or equal to max
        if min_dist_range > max_dist_range:
            msg = (f'Minimum range ratio {min_dist_range} is larger'
                   f' than maximum range ratio {max_dist_range}')
            raise ValueError(msg)

        # Check if min is negative
        if min_dist_range <= 0:
            msg = (f'Minimum range ratio is below or equal to '
                   f'0 and it is {min_dist_range}')
            raise ValueError(msg)

        # Check if max is larger than 1
        if max_dist_range > 1:
            msg = (f'Maximum range ratio should be lower than 1, '
                   f'but it is {max_dist_range}')
            raise ValueError(msg)

    @staticmethod
    def __validate_nugget_ranges(min_nugget: float, max_nugget: float):
        # Check if min is lower or equal to max
        if min_nugget > max_nugget:
            msg = (f'The minimum nugget {min_nugget} is larger than the '
                   f'maximum nugget {max_nugget}')
            raise ValueError(msg)

        # Check if min is negative
        if min_nugget < 0:
            msg = (f'Minimum nugget is lower than 0 and it'
                   f'is equal to {min_nugget}')
            raise ValueError(msg)

    @staticmethod
    def __validate_sill_ranges(min_sill: float, max_sill: float):
        # Check if min is lower or equal to max
        if min_sill > max_sill:
            msg = (f'Minimum sill ratio {min_sill} is larger '
                   f'than maximum sill ratio {max_sill}')
            raise ValueError(msg)

        # Check if min is negative
        if min_sill < 0:
            msg = (f'Minimum sill ratio is below '
                   f'0 and it is equal to {min_sill}')
            raise ValueError(msg)

    def _params_are_given(self):
        if (
                self.model_type is not None
        ) and (
                self.nugget is not None
        ) and (
                self.sill is not None
        ) and (
                self.rang is not None
        ):
            return True

    def _plot_from_params(self):
        legend = ['Theoretical Model']
        plt.figure(figsize=(12, 6))

        lags = np.linspace(0, self.rang * 5, 50)
        yhat = self.predict(lags)
        plt.plot(lags, yhat, '--', color='#fc8d62')
        plt.legend(legend)
        plt.xlabel('Distance')
        plt.ylabel('Variance')
        plt.show()
