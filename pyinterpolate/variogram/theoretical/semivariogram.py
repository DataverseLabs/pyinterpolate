"""
TheoreticalVariogram class.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

Contributors
------------
1. Ethem Turgut | @ethmtrgt
"""
# Core python packages
import json
import warnings
from typing import Collection, Union, Callable, Tuple

# Core calculations and visualization packages
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Pyinterpolate dependencies
from pyinterpolate.processing.select_values import create_min_max_array, get_study_max_range
from pyinterpolate.variogram.theoretical.models import circular_model, cubic_model, linear_model, exponential_model, \
    gaussian_model, spherical_model, power_model
from pyinterpolate.variogram.empirical.experimental_variogram import ExperimentalVariogram
from pyinterpolate.variogram.utils.metrics import forecast_bias, root_mean_squared_error, \
    symmetric_mean_absolute_percentage_error, mean_absolute_error, weighted_root_mean_squared_error
from pyinterpolate.variogram.utils.exceptions import validate_selected_errors, check_ranges, check_sills


class TheoreticalVariogram:
    """Theoretical model of a spatial dissimilarity.

    Parameters
    ----------
    model_params : Dict, default=None
        Dictionary with ``'nugget'``, ``'sill'``, ``'range'`` and ``'name'`` of the model.

    Attributes
    ----------
    experimental_variogram : EmpiricalVariogram, default=None
        Empirical Variogram class and its attributes.

    experimental_array : numpy array, default=None
        Empirical variogram in a form of numpy array.

    variogram_models : Dict
        A dictionary with keys representing theoretical variogram models and values that are pointing into a modeling
        functions. Available models:

        - 'circular',
        - 'cubic',
        - 'exponential',
        - 'gaussian',
        - 'linear',
        - 'power',
        - 'spherical'.

    fitted_model : numpy array or None
        Trained theoretical model values. Array of ``[lag, variances]``.

    name : str or None, default=None
        The name of the chosen model. Available names are the same as keys in ``variogram_models`` attribute.

    nugget : float, default=0
        The nugget parameter (bias at a zero distance).

    sill : float, default=0
        A value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
        to observations variance.

    rang : float, default=0
        The semivariogram range is a distance at which spatial correlation exists. It shouldn't be set at a distance
        larger than a half of a study extent.

    direction : float, in range [0, 360], default=None
        The direction of a semivariogram. If ``None`` given then semivariogram is isotropic.

    rmse : float, default=0
        Root mean squared error of the difference between the empirical observations and the modeled curve.

    mae : bool, default=True
        Mean Absolute Error of a model.

    bias : float, default=0
        Forecast Bias of the modeled variogram vs experimental points. Large positive value means that the estimated
        model underestimates values. A large negative value means that model overestimates predictions.

    smape : float, default=0
        Symmetric Mean Absolute Percentage Error of the prediction - values from 0 to 100%.

    are_params : bool
        Check if model parameters were given during initialization.

    Methods
    -------
    fit()
        Fits experimental variogram data into theoretical model.

    autofit()
        The same as fit but tests multiple ranges, sills and models.

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
    >>> import numpy
    >>> REFERENCE_INPUT = numpy.array([
    ...    [0, 0, 1],
    ...    [1, 0, 2],
    ...    [2, 0, 3],
    ...    [3, 0, 4],
    ...    [4, 0, 5],
    ...    [5, 0, 6],
    ...    [6, 0, 7],
    ...    [7, 0, 5],
    ...    [8, 0, 3],
    ...    [9, 0, 1],
    ...    [10, 0, 4],
    ...    [11, 0, 6],
    ...    [12, 0, 8]
    ...    ])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = ExperimentalVariogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> theoretical_smv = TheoreticalVariogram()
    >>> _ = theoretical_smv.autofit(experimental_variogram=empirical_smv, model_types='gaussian')
    >>> print(theoretical_smv.rmse)
    1.5275214898546217
    """

    def __init__(self, model_params: Union[dict, None] = None):

        self.are_params = isinstance(model_params, dict)

        # Model
        self.variogram_models = {
            'circular': circular_model,
            'cubic': cubic_model,
            'exponential': exponential_model,
            'gaussian': gaussian_model,
            'linear': linear_model,
            'power': power_model,
            'spherical': spherical_model
        }
        self.study_max_range = None

        # Model parameters
        self.lags = None
        self.experimental_variogram = None
        self.experimental_array = None
        self.fitted_model = None

        self.name = None
        self.nugget = 0.
        self.rang = 0.
        self.sill = 0.

        self.direction = None

        if self.are_params:
            self._set_model_parameters(model_params)

        # Errror
        self.deviation_weighting = None
        self.rmse = 0.
        self.bias = 0.
        self.smape = 0.
        self.mae = 0.

    # Core functions

    def fit(self,
            experimental_variogram: Union[ExperimentalVariogram, np.ndarray],
            model_type: str,
            sill: float,
            rang: float,
            nugget=0.,
            direction=None,
            update_attrs=True,
            warn_about_set_params=True) -> Tuple[np.array, dict]:
        """

        Parameters
        ----------
        experimental_variogram : ExperimentalVariogram
            Prepared Empirical Variogram.

        model_type : str
            Model type. Available models:

            - 'circular',
            - 'cubic',
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical'.

        sill : float, default=0
            A value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
            to observations variance.

        rang : float, default=0
            The semivariogram range is a distance at which spatial correlation exists. It shouldn't be set at a distance
            larger than a half of a study extent.

        nugget : float, default=0.
            Nugget parameter (bias at a zero distance).

        direction : float, in range [0, 360], default=None
            The direction of a semivariogram. If ``None`` given then semivariogram is isotropic. This parameter is
            required if passed experimental variogram is stored in a numpy array.

        update_attrs : bool, default=True
            Should class attributes be updated?

        warn_about_set_params: bool, default=True
            Should class invoke warning if model parameters has been set during initialization?

        Raises
        ------
        KeyError
            Model type (function) not implemented.

        Warns
        -----
        Warning
            Model parameters were given during initialization but program is forced to fit the new set of parameters.

        Warning
            Passed ``experimental_variogram`` is a numpy array and ``direction`` parameter is ``None``.

        Returns
        -------
        _theoretical_values, _error: Tuple[ numpy array, dict ]
            ``[ theoretical semivariances, {'rmse bias smape mae'}]``

        """
        if self.are_params:
            if warn_about_set_params:
                warnings.warn('Semivariogram parameters have been set earlier, you are going to overwrite them')

        if isinstance(experimental_variogram, ExperimentalVariogram):
            self.experimental_variogram = experimental_variogram
            self.lags = experimental_variogram.lags
            self.experimental_array = experimental_variogram.experimental_semivariance_array
            self.direction = direction
        elif isinstance(experimental_variogram, np.ndarray):
            self.experimental_array = experimental_variogram
            self.lags = experimental_variogram[:, 0]
            if direction is None:
                msg = 'If you provide experimental variogram as a numpy array you must remember that the direction' \
                      ' parameter must be set if it is a directional variogram. Otherwise, algorithm assumes that' \
                      ' variogram is isotropic.'
                warnings.warn(msg)
            self.direction = direction

        # Check model type

        _names = list(self.variogram_models.keys())

        if model_type not in _names:
            msg = f'Defined model name {model_type} not available. You may choose one from {_names} instead.'
            raise KeyError(msg)

        _model = self.variogram_models[model_type]

        _theoretical_values = self._fit_model(_model, nugget, sill, rang)

        # Estimate errors
        _error = self.calculate_model_error(_theoretical_values[:, 1], rmse=True, bias=True, smape=True)

        if update_attrs:
            attrs_to_update = {
                'fitted_model': _theoretical_values,
                'model_type': model_type,
                'nugget': nugget,
                'sill': sill,
                'rang': rang
            }
            attrs_to_update.update(_error)

            self._update_attributes(**attrs_to_update)

        return _theoretical_values, _error

    def autofit(self,
                experimental_variogram: Union[ExperimentalVariogram, np.ndarray],
                model_types: Union[str, list] = 'all',
                nugget=0,
                rang=None,
                min_range=0.1,
                max_range=0.5,
                number_of_ranges=64,
                sill=None,
                min_sill=0.,
                max_sill=1,
                number_of_sills=64,
                direction=None,
                error_estimator='rmse',
                deviation_weighting='equal',
                auto_update_attributes=True,
                warn_about_set_params=True,
                verbose=False,
                return_params=True):
        """
        Method tries to find the optimal range, sill and model (function) of the theoretical semivariogram.

        Parameters
        ----------
        experimental_variogram : ExperimentalVariogram
            Prepared Empirical Variogram or array.

        model_types : str or list
            List of modeling functions or a name of a single function. Available models:

            - 'all' - the same as list with all models,
            - 'circular',
            - 'cubic',
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical'.

        nugget : float, default = 0
            Nugget (bias) of a variogram. Default value is 0.

        rang : float, optional
            If given, then range is fixed to this value.

        min_range : float, default = 0.1
            The minimal fraction of a variogram range, ``0 < min_range <= max_range``.

        max_range : float, default = 0.5
            The maximum fraction of a variogram range, ``min_range <= max_range <= 1``. Parameter ``max_range`` greater
            than **0.5** raises warning.

        number_of_ranges : int, default = 64
            How many equally spaced ranges are tested between ``min_range`` and ``max_range``.

        sill : float, default = None
            If given, then sill is fixed to this value.

        min_sill : float, default = 0
            The minimal fraction of the variogram variance at lag 0 to find a sill, ``0 <= min_sill <= max_sill``.

        max_sill : float, default = 1
            The maximum fraction of the variogram variance at lag 0 to find a sill. It *should be* lower or equal to 1.
            It is possible to set it above 1, but then warning is printed.

        number_of_sills : int, default = 64
            How many equally spaced sill values are tested between ``min_sill`` and ``max_sill``.

        direction : float, in range [0, 360], default=None
            The direction of a semivariogram. If ``None`` given then semivariogram is isotropic. This parameter is
            required if passed experimental variogram is stored in a numpy array.

        error_estimator : str, default = 'rmse'
            A model error estimation method. Available options are:

            - 'rmse': Root Mean Squared Error,
            - 'mae': Mean Absolute Error,
            - 'bias': Forecast Bias,
            - 'smape': Symmetric Mean Absolute Percentage Error.

        deviation_weighting : str, default = "equal"
            The name of a method used to weight error at a given lags. Works only with RMSE. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger weights,
            - distant: lags that are further away have bigger weights,
            - dense: error is weighted by the number of point pairs within a lag.

        auto_update_attributes : bool, default = True
            Update sill, range, model type and nugget based on the best model.

        warn_about_set_params: bool, default=True
            Should class invoke warning if model parameters has been set during initialization?

        verbose : bool, default = False
            Show iteration results.

        return_params : bool, default = True
            Return model parameters.


        Returns
        -------
        model_attributes : Dict
            Attributes dict:

            >>> {
            ...     'model_type': model_name,
            ...     'sill': model_sill,
            ...     'rang': model_range,
            ...     'nugget': model_nugget,
            ...     'error_type': type_of_error_metrics,
            ...     'error value': error_value
            ... }

        Warns
        -----
        SillOutsideSafeRangeWarning
            Warning printed when ``max_sill > 1``.

        RangeOutsideSafeDistanceWarning
            Warning printed when ``max_range > 0.5``.

        Warning
            Model parameters were given during initilization but program is forced to fit the new set of parameters.

        Warning
            Passed ``experimental_variogram`` is a numpy array and ``direction`` parameter is ``None``.

        Raises
        ------
        ValueError
            Raised when ``sill < 0`` or ``range < 0`` or ``range > 1``.

        KeyError
            Raised when wrong model name(s) are provided by the users.

        KeyError
            Raised when wrong error type is provided by the users.

        TODO
        ----
        * add 'safe' models list to autofit() method
        """

        self.deviation_weighting = deviation_weighting

        if self.are_params:
            if warn_about_set_params:
                warnings.warn('Semivariogram parameters have been set earlier, you are going to overwrite them')

        if isinstance(experimental_variogram, ExperimentalVariogram):
            self.experimental_variogram = experimental_variogram
            self.lags = experimental_variogram.lags
            self.experimental_array = experimental_variogram.experimental_semivariance_array
            self.direction = experimental_variogram.direct
        elif isinstance(experimental_variogram, np.ndarray):
            self.experimental_array = experimental_variogram
            self.lags = experimental_variogram[:, 0]
            if direction is None:
                msg = 'If you provide experimental variogram as a numpy array you must remember that the direction' \
                      ' parameter must be set if it is a directional variogram. Otherwise, algorithm assumes that' \
                      ' variogram is isotropic.'
                warnings.warn(msg)
            self.direction = direction

        # Check model type and set models
        mtypes = self._check_models_type_autofit(model_types)

        # Set ranges and sills
        if rang is None:
            check_ranges(min_range, max_range)
            if self.study_max_range is None:
                if self.experimental_variogram is not None:
                    self.study_max_range = get_study_max_range(self.experimental_variogram.input_array[:, :-1])
                else:
                    self.study_max_range = self.experimental_array[-1, 0]

            min_max_ranges = create_min_max_array(self.study_max_range, min_range, max_range, number_of_ranges)
        else:
            min_max_ranges = [rang]

        if sill is None:
            check_sills(min_sill, max_sill)
            if self.experimental_variogram is not None:
                var_sill = self.experimental_variogram.variance
            else:
                ll = len(self.experimental_array)
                pos = int(0.2 * ll)
                var_sill = np.mean(self.experimental_array[-pos:, 1])

            min_max_sill = create_min_max_array(var_sill, min_sill, max_sill, number_of_sills)
        else:
            min_max_sill = [sill]

        # Get errors
        _errors_keys = self._get_err_keys(error_estimator)

        # Initialize error
        err_val = np.inf

        # Initlize parameters
        optimal_parameters = {
            'model_type': '',
            'nugget': 0,
            'sill': 0,
            'rang': 0
        }

        for _mtype in mtypes:
            for _rang in min_max_ranges:
                for _sill in min_max_sill:
                    # Create model
                    _mdl_fn = self.variogram_models[_mtype]
                    _fitted_model = self._fit_model(_mdl_fn, nugget, _sill, _rang)

                    # Calculate Error
                    _err = self.calculate_model_error(_fitted_model[:, 1],
                                                      **_errors_keys,
                                                      deviation_weighting=deviation_weighting)

                    if verbose:
                        self.__print_autofit_info(_mtype, nugget, _sill, _rang, error_estimator, _err[error_estimator])

                    # Check if model is better than the previous
                    if _err[error_estimator] < err_val:
                        err_val = _err[error_estimator]
                        optimal_parameters['model_type'] = _mtype
                        optimal_parameters['nugget'] = nugget
                        optimal_parameters['sill'] = _sill
                        optimal_parameters['rang'] = _rang
                        optimal_parameters['fitted_model'] = _fitted_model
                        optimal_parameters.update(_err)

        if auto_update_attributes:
            self._update_attributes(**optimal_parameters)

        if return_params:
            return optimal_parameters

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """
        Method returns a semivariance per distance.

        Parameters
        ----------
        distances : np.ndarray

        Returns
        -------
        predicted : np.ndarray

        """

        _model = self.variogram_models[self.name]
        predicted = _model(
            distances, self.nugget, self.sill, self.rang
        )

        return predicted

    # Plotting and visualization

    def plot(self, experimental=True):
        """
        Method plots theoretical curve and (optionally) experimental scatter plot.

        Parameters
        ----------
        experimental : bool
            Plot experimental observations.

        Raises
        ------
        AttributeError
            Model is not fitted yet, nothing to plot.
        """
        if self.fitted_model is None:
            raise AttributeError('Model has not been trained, nothing to plot.')
        else:
            legend = ['Theoretical Model']
            plt.figure(figsize=(12, 6))

            if experimental:
                plt.scatter(self.experimental_array[:, 0],
                            self.experimental_array[:, 1],
                            marker='8', c='#66c2a5')
                legend = ['Experimental Semivariances', 'Theoretical Model']

            plt.plot(self.lags, self.fitted_model[:, 1], '--', color='#fc8d62')
            plt.legend(legend)
            plt.xlabel('Distance')
            plt.ylabel('Variance')
            plt.show()

    # Evaluation

    def calculate_model_error(self,
                              fitted_values: np.array,
                              rmse=True,
                              bias=True,
                              mae=True,
                              smape=True,
                              deviation_weighting='equal') -> dict:
        """
        Method calculates error associated with a difference between the theoretical model and
        the experimental semivariogram.

        Parameters
        ----------
        fitted_values : numpy array
            ``[lag, fitted value]``

        rmse : bool, default=True
            Root Mean Squared Error of a model.

        bias : bool, default=True
            Forecast Bias of a model.

        mae : bool, default=True
            Mean Absolute Error of a model.

        smape : bool, default=True
            Symmetric Mean Absolute Percentage Error of a model.

        deviation_weighting : str, default = "equal"
            The name of a method used to **weight errors at a given lags**. Works only with RMSE.
            Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger weights,
            - distant: lags that are further away have bigger weights,
            - dense: error is weighted by the number of point pairs within a lag.

        Returns
        -------
        model_errors : Dict
            Dict with error values per model: rmse, bias, mae, smape.

        Raises
        ------
        MetricsTypeSelectionError
            User has set all error types to ``False``.
        """

        # Check errors
        validate_selected_errors(rmse + bias + mae + smape)  # all False sums to 0 -> error detection
        model_error = {
            'rmse': np.nan,
            'bias': np.nan,
            'mae': np.nan,
            'smape': np.nan
        }

        _real_values = self.experimental_array[:, 1].copy()

        # Get Forecast Biast
        if bias:
            _fb = forecast_bias(fitted_values, _real_values)
            model_error['bias'] = _fb

        # Get RMSE
        if rmse:
            if deviation_weighting != 'equal':
                if deviation_weighting == 'dense':
                    points_per_lag = self.experimental_array[:, -1]
                    _rmse = weighted_root_mean_squared_error(fitted_values, _real_values, deviation_weighting,
                                                             lag_points_distribution=points_per_lag)
                else:
                    _rmse = weighted_root_mean_squared_error(fitted_values, _real_values, deviation_weighting)
            else:
                _rmse = root_mean_squared_error(fitted_values, _real_values)
            model_error['rmse'] = _rmse

        # Get MAE
        if mae:
            _mae = mean_absolute_error(fitted_values, _real_values)
            model_error['mae'] = _mae

        # Get SMAPE
        if smape:
            _smape = symmetric_mean_absolute_percentage_error(fitted_values, _real_values)
            model_error['smape'] = _smape

        return model_error

    # I/O

    def to_dict(self) -> dict:
        """Method exports the theoretical variogram parameters to a dictionary.

        Returns
        -------
        model_parameters : Dict
            Dictionary with model's ``'name'``, ``'nugget'``, ``'sill'``, ``'range'`` and ``'direction'``.

        Raises
        ------
        AttributeError
            The model parameters have not been derived yet.
        """

        if self.fitted_model is None:
            if self.name is None:
                msg = 'Model is not set yet, cannot export model parameters to dict'
                raise AttributeError(msg)

        modeled_parameters = {
            'name': self.name,
            'sill': self.sill,
            'range': self.rang,
            'nugget': self.nugget,
            'direction': self.direction
        }

        return modeled_parameters

    def from_dict(self, parameters: dict) -> None:
        """Method updates model with a given set of parameters.

        Parameters
        ----------
        parameters : Dict
            Dictionary with model's: ``'name', 'nugget', 'sill', 'range', 'direction'``.
        """

        self._set_model_parameters(parameters)
        self.are_params = True

    def to_json(self, fname: str):
        """
        Method stores variogram parameters into a JSON file.

        Parameters
        ----------
        fname : str
        """

        json_output = {
            'name': self.name,
            'nugget': self.nugget,
            'range': self.rang,
            'sill': self.sill,
            'direction': self.direction
        }

        with open(fname, 'w') as fout:
            json.dump(json_output, fout)

    def from_json(self, fname: str):
        """
        Method reads data from a JSON file.

        Parameters
        ----------
        fname : str
        """

        with open(fname, 'r') as fin:
            json_input = json.load(fin)

        self._set_model_parameters(json_input)
        self.are_params = True

    def __str__(self):

        check_validity = (self.name is None) and (self.rang == 0) and (self.sill == 0)

        if check_validity:
            return 'Theoretical model is not calculated yet. Use fit() or autofit() methods to build or find a model ' \
                   'or import model with from_dict() or from_json() methods.'
        else:
            title = '* Selected model: ' + f'{self.name}'.capitalize() + ' model'
            msg_nugget = f'* Nugget: {self.nugget}'
            msg_sill = f'* Sill: {self.sill}'
            msg_range = f'* Range: {self.rang}'
            mean_bias_msg = f'* Mean Bias: {self.bias}'
            mean_rmse_msg = f'* Mean RMSE: {self.rmse}'
            error_weighting = f'* Error-lag weighting method: {self.deviation_weighting}'

            text_list = [title, msg_nugget, msg_sill, msg_range, mean_bias_msg, mean_rmse_msg, error_weighting]

            header = '\n'.join(text_list) + '\n' + '\n'

            if self.experimental_array is not None:
                # Build pretty table
                pretty_table = PrettyTable()
                pretty_table.field_names = ["lag", "theoretical", "experimental", "bias (y-y')"]

                records = []
                for idx, record in enumerate(self.experimental_array):
                    lag = record[0]
                    experimental_semivar = record[1]
                    theoretical_semivar = self.fitted_model[idx][1]
                    bias = experimental_semivar - theoretical_semivar
                    records.append([lag, theoretical_semivar, experimental_semivar, bias])

                pretty_table.add_rows(records)

                msg = header + pretty_table.get_string()
                return msg
            else:
                return header

    def __repr__(self):
        cname = 'TheoreticalVariogram'
        input_params = f'empirical_variogram={self.experimental_variogram},experimental_array={self.experimental_array}'
        repr_val = cname + '(' + input_params + ')'
        return repr_val

    def _check_model_names(self, mname):
        _names = list(self.variogram_models.keys())

        if mname not in _names:
            msg = f'Defined model name {mname} not available. You may choose one from {_names} instead.'
            raise KeyError(msg)

    def _check_models_type_autofit(self, model_types: Union[str, Collection]) -> list:
        mtypes = list()
        if isinstance(model_types, str):
            if model_types == 'all':
                mtypes = [
                    'circular',
                    'cubic',
                    'exponential',
                    'gaussian',
                    'linear',
                    'power',
                    'spherical'
                ]
            else:
                self._check_model_names(model_types)
                mtypes.append(model_types)
        else:
            if isinstance(model_types, Collection):
                for mdl in model_types:
                    try:
                        self._check_model_names(mdl)
                    except KeyError as ke:
                        if mdl == 'all' and len(model_types) == 1:
                            mtypes = [
                                'circular',
                                'cubic',
                                'exponential',
                                'gaussian',
                                'linear',
                                'power',
                                'spherical'
                            ]
                        else:
                            raise ke
                    mtypes.append(mdl)
        return mtypes

    def _update_attributes(self,
                           fitted_model=None,
                           model_type=None,
                           nugget=None,
                           rang=None,
                           sill=None,
                           rmse=None,
                           bias=None,
                           mae=None,
                           smape=None):
        # Model parameters
        self.fitted_model = fitted_model
        self.name = model_type
        self.nugget = nugget
        self.rang = rang
        self.sill = sill

        # Dynamic parameters
        self.rmse = rmse
        self.bias = bias
        self.smape = smape
        self.mae = mae

    @staticmethod
    def _get_err_keys(err_name: str):
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
            msg = f'Defined error {err_name} not exists. Use one of {list(err_dict.keys())} instead.'
            raise KeyError(msg)

    @staticmethod
    def __print_autofit_info(model_type: str, nugget: float, sill: float, rang: float, err_type: str, err_value: float):
        msg_core = f'Model {model_type},\n' \
                   f'Model Parameters - nugget: {nugget:.2f}, sill: {sill:.4f}, range: {rang:.4f},\n' \
                   f'Model Error {err_type}: {err_value}' \
                   f'\n'
        print(msg_core)

    def _fit_model(self, model_fn: Callable, nugget: float, sill: float, rang: float) -> np.array:
        """Method fits selected model into baseline lags.

        Parameters
        ----------
        model_fn : Callable
                   Selected model.

        nugget : float

        sill : float

        rang : float

        Returns
        -------
        : numpy array
        """

        lags = self.experimental_array[:, 0]
        fitted_values = model_fn(lags, nugget, sill, rang)
        modeled = np.zeros(shape=(len(lags), 2))
        modeled[:, 0] = lags
        modeled[:, 1] = fitted_values
        return modeled

    @staticmethod
    def _set_direction(mparams):
        try:
            direction = float(mparams['direction'])
        except TypeError:
            return None
        except KeyError:
            return None

        return direction

    def _set_model_parameters(self, model_params: dict):
        self.nugget = float(model_params['nugget'])
        self.rang = float(model_params['range'])
        self.sill = float(model_params['sill'])
        self.name = model_params['name']
        self.direction = self._set_direction(model_params)


def build_theoretical_variogram(experimental_variogram: ExperimentalVariogram,
                                model_type: str,
                                sill: float,
                                rang: float,
                                nugget: float = 0.,
                                direction: float = None) -> TheoreticalVariogram:
    """Function is a wrapper into ``TheoreticalVariogram`` class and its ``fit()`` method.

    Parameters
    ----------
    experimental_variogram : ExperimentalVariogram

    model_type : str
        Available types:

        - 'circular',
        - 'cubic',
        - 'exponential',
        - 'gaussian',
        - 'linear',
        - 'power',
        - 'spherical'.

    sill : float
        The value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
        to observations variance.

    rang : float
        The semivariogram range is a distance at which spatial correlation exists. It shouldn't be set at a distance
        larger than a half of a study extent.

    nugget : float, default=0.
        The nugget parameter (bias at a zero distance).

    direction : float, in range [0, 360], default=None
        The direction of a semivariogram. If ``None`` given then semivariogram is isotropic.

    Returns
    -------
    theo : TheoreticalVariogram
        Fitted theoretical semivariogram model.
    """
    theo = TheoreticalVariogram()
    theo.fit(
        experimental_variogram=experimental_variogram,
        model_type=model_type,
        sill=sill,
        rang=rang,
        nugget=nugget,
        direction=direction
    )
    return theo
