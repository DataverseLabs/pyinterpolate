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
from pyinterpolate.variogram.empirical import ExperimentalVariogram
from pyinterpolate.variogram.utils.metrics import forecast_bias, root_mean_squared_error, \
    symmetric_mean_absolute_percentage_error, mean_absolute_error
from pyinterpolate.variogram.utils.exceptions import validate_selected_errors, check_ranges, check_sills


class TheoreticalVariogram:
    """Theoretical model of spatial data dissimilarity.

    Parameters
    ----------
    model_params : dict or None, default=None
                   Dictionary with 'nugget', 'sill', 'range' and 'name' of the model.

    Attributes
    ----------
    empirical_variogram : EmpiricalVariogram or None, default=None
                          Empirical Variogram class and its attributes.

    variogram_models : dict
                       Dict with keys representing theoretical variogram models and values that
                       are pointing into a modeling methods. Available models:
                           'circular',
                           'cubic',
                           'exponential',
                           'gaussian',
                           'linear',
                           'power',
                           'spherical'.

    fitted_model : numpy array or None
                   Trained theoretical values model. Array of [lag, variances].

    name : str or None, default=None
           Name of the chosen model. Available names are the same as keys in variogram_models attribute.

    nugget : float, default=0
             Nugget parameter (bias at a zero distance).

    sill : float, default=0
           Value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
           to observations variance.

    rang : float, default=0
          Semivariogram Range is a distance at which spatial correlation exists and often it is a distance when
          variogram reaches sill. It shouldn't be set at a distance larger than a half of a study extent.

    rmse : float, default=0
           Root mean squared error of the difference between the empirical observations and the modeled curve.

    mae : bool, default=True
          Mean Absolute Error of a model.

    bias : float, default=0
           Forecast Bias of the estimation. Large positive value means that the estimated model usually underestimates
           values and large negative value means that model overestimates predictions.

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

    __str__()
        Prints basic info about the class parameters.

    __repr__()
        Reproduces class initialization with an input experimental variogram.

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
    >>> _ = theoretical_smv.autofit(empirical_variogram=empirical_smv, model_types='gaussian')
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
        self.empirical_variogram = None
        self.fitted_model = None

        self.name = None
        self.nugget = 0.
        self.rang = 0.
        self.sill = 0.

        if self.are_params:
            self._set_model_parameters(model_params)

        # Dynamic parameters
        self.rmse = 0.
        self.bias = 0.
        self.smape = 0.
        self.mae = 0.

    # Core functions

    def fit(self,
            empirical_variogram: ExperimentalVariogram,
            model_type: str,
            sill: float,
            rang: float,
            nugget=0.,
            update_attrs=True,
            warn_about_set_params=True) -> Tuple[np.array, dict]:
        """

        Parameters
        ----------
        empirical_variogram : ExperimentalVariogram
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

        sill : float
               Value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
               to observations variance.

        rang : float, default=0
                 Semivariogram Range is a distance at which spatial correlation exists and often it is a distance when
                 variogram reaches sill. It shouldn't be set at a distance larger than a half of a study extent.

        nugget : float, default=0.
                 Nugget parameter (bias at a zero distance).

        update_attrs : bool, default=True
                       Should class attributes be updated?

        warn_about_set_params: bool, default=True
                               Should class invoke warning if model parameters has been set during initialization?

        Raises
        ------
        KeyError : Model type not implemented

        Warns
        -----
        : Model parameters were given during initilization but program is forced to fit the new set of parameters.

        Returns
        -------
        : Tuple[ numpy array, dict ]
            [ theoretical semivariances, {'rmse bias smape mae'}]

        """
        if self.are_params:
            if warn_about_set_params:
                warnings.warn('Semivariogram parameters have been set earlier, you are going to overwrite them')

        self.empirical_variogram = empirical_variogram
        self.lags = empirical_variogram.lags

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
                empirical_variogram: ExperimentalVariogram,
                model_types: Union[str, list],
                nugget=0,
                min_range=0.1,
                max_range=0.5,
                number_of_ranges=16,
                min_sill=0.,
                max_sill=1,
                number_of_sills=16,
                error_estimator='rmse',
                auto_update_attributes=True,
                warn_about_set_params=True,
                verbose=False):
        """
        Methodtries to find the optimal range, sill and model of theoretical semivariogram.

        Parameters
        ----------
        empirical_variogram : ExperimentalVariogram
                          Prepared Empirical Variogram.

        model_types : str or list
                      List of models of string with a model name. Available models:
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

        min_range : float, default = 0.1
                    Minimal fraction of a variogram range, 0 < min_range <= max_range

        max_range : float, default = 0.5
                    Maximum fraction of a variogram range, min_range <= max_range <= 1. Parameter max_range greater
                    than 0.5 raises warning.

        number_of_ranges : int, default = 16
                           How many equally spaced ranges are tested between min_range and max_range.

        min_sill : float, default = 0
                   Minimal fraction of variogram variance at lag 0 to find a sill, 0 <= min_sill <= max_sill.

        max_sill : float, default = 0
                   Maximum fraction of variogram variance at lag 0 to find a sill. It should be lower or equal to 1. but
                   it is possible to set it above. Warning is printed if max_sill is greater than 1,
                   min_sill <= max_sill.

        number_of_sills : int, default = 16
                          How many equally spaced sill values are tested between min_sill and max_sill.

        error_estimator : str, default = 'rmse'
                          Error estimation to choose the best model. Available options are:
                          - 'rmse': Root Mean Squared Error,
                          - 'mae': Mean Absolute Error,
                          - 'bias': Forecast Bias,
                          - 'smape': Symmetric Mean Absolute Percentage Error.

        auto_update_attributes : bool, default = True
                                 Update sill, range, model type and nugget based on the best model.

        warn_about_set_params: bool, default=True
                               Should class invoke warning if model parameters has been set during initialization?

        verbose : bool, default = False
                  Show iteration results.


        Returns
        -------
        model_attributes : dict
                           {
                                'model_type': model name,
                                'sill': model sill,
                                'rang': model range,
                                'nugget': model nugget,
                                'error_type': type of error metrics,
                                'error value': error value
                           }

        Warns
        -----
        SillOutsideSafeRangeWarning : max_sill > 1

        RangeOutsideSafeDistanceWarning : max_range > 0.5

        : Model parameters were given during initilization but program is forced to fit the new set of parameters.

        Raises
        ------
        ValueError : sill < 0 or range < 0 or range > 1.

        KeyError : wrong model name(s) or wrong error type name.
        """

        if self.are_params:
            if warn_about_set_params:
                warnings.warn('Semivariogram parameters have been set earlier, you are going to overwrite them')

        self.empirical_variogram = empirical_variogram
        self.lags = empirical_variogram.lags

        # Check parameters
        check_ranges(min_range, max_range)
        check_sills(min_sill, max_sill)

        # Check model type and set models
        mtypes = self._check_models_type_autofit(model_types)

        # Set ranges and sills
        if self.study_max_range is None:
            self.study_max_range = get_study_max_range(self.empirical_variogram.input_array[:, :-1])

        min_max_ranges = create_min_max_array(self.study_max_range, min_range, max_range, number_of_ranges)

        var_sill = self.empirical_variogram.variance
        min_max_sill = create_min_max_array(var_sill, min_sill, max_sill, number_of_sills)

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
                    _err = self.calculate_model_error(_fitted_model[:, 1], **_errors_keys)

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

        return optimal_parameters

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """
        Method returns semivariances for a given distances.

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
        Method plots theoretical curve and (optionally) experimental scatterplot.

        Parameters
        ----------
        experimental : bool
                       Plot experimental observations.

        Raises
        ------
        AttributeError : Model is not fitted yet
        """
        if self.fitted_model is None:
            raise AttributeError('Model has not been trained, nothing to plot.')
        else:
            legend = []
            plt.figure(figsize=(12, 6))

            if experimental:
                plt.scatter(self.lags,
                            self.empirical_variogram.experimental_semivariances,
                            marker='8', c='#66c2a5')
                legend.append('Experimental Semivariances')

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
                              smape=True) -> dict:
        """

        Parameters
        ----------
        fitted_values : numpy array
                        [lag, fitted value]

        rmse : bool, default=True
               Root Mean Squared Error of a model.

        bias : bool, default=True
               Forecast Bias of a model.

        mae : bool, default=True
              Mean Absolute Error of a model.

        smape : bool, default=True
                Symmetric Mean Absolute Percentage Error of a model.

        Returns
        -------
        model_errors : dict
                       Dict with error values per model: rmse, bias, mae, smape.

        Raises
        ------
        MetricsTypeSelectionError : User set all errors to False
        """

        # Check errors
        validate_selected_errors(rmse + bias + mae + smape)  # all False sums to 0 -> error detection
        model_error = {
            'rmse': np.nan,
            'bias': np.nan,
            'mae': np.nan,
            'smape': np.nan
        }

        _real_values = self.empirical_variogram.experimental_semivariance_array[:, 1].copy()

        # Get Forecast Biast
        if bias:
            _fb = forecast_bias(fitted_values, _real_values)
            model_error['bias'] = _fb

        # Get RMSE
        if rmse:
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
        """Method exports theoretical variogram parameters to dictionary.

        Returns
        -------
        model_parameters : dict
                           Dictionary with model 'name', 'nugget', 'sill' and 'range'.

        Raises
        ------
        AttributeError : Model parameters are not derived yet
        """

        if self.fitted_model is None:
            if self.name is None:
                msg = 'Model is not set yet, cannot export model parameters to dict'
                raise AttributeError(msg)

        modeled_parameters = {'name': self.name,
                              'sill': self.sill,
                              'range': self.rang,
                              'nugget': self.nugget}

        return modeled_parameters

    def from_dict(self, parameters: dict) -> None:
        """Method updates model with a given parameters.

        Parameters
        ----------
        parameters : dict
                     'name', 'nugget', 'sill', 'range'
        """

        self._set_model_parameters(parameters)
        self.are_params = True

    def to_json(self, fname: str):
        """
        Method stores variogram parameters into a JSON file.

        Parameters
        ----------
        fname : str
                File to store a data.

        """

        json_output = {
            'name': self.name,
            'nugget': self.nugget,
            'range': self.rang,
            'sill': self.sill
        }

        with open(fname, 'w') as fout:
            json.dump(json_output, fout)

    def from_json(self, fname: str):
        """
        Method reads data from a JSON file.

        Parameters
        ----------
        fname : str
                File with a stored parameters.
        """

        with open(fname, 'r') as fin:
            json_input = json.load(fin)

        self._set_model_parameters(json_input)
        self.are_params = True

    def __str__(self):

        if self.fitted_model is None:
            return 'Theoretical model is not calculated yet. Use fit() or autofit() methods to build or find a model.'
        else:
            title = '* Selected model: ' + f'{self.name}'.capitalize() + ' model'
            msg_nugget = f'* Nugget: {self.nugget}'
            msg_sill = f'* Sill: {self.sill}'
            msg_range = f'* Range: {self.rang}'
            mean_bias_msg = f'* Mean Bias: {self.bias}'
            mean_rmse_msg = f'* Mean RMSE: {self.rmse}'

            text_list = [title, msg_nugget, msg_sill, msg_range, mean_bias_msg, mean_rmse_msg]

            header = '\n'.join(text_list) + '\n'

            # Build pretty table
            pretty_table = PrettyTable()
            pretty_table.field_names = ["lag", "experimental", "theoretical", "bias", "rmse"]

            records = []
            for idx, record in enumerate(self.empirical_variogram.experimental_semivariance_array):
                lag = record[0]
                experimental_semivar = record[1]
                theoretical_semivar = self.fitted_model[idx][1]
                bias = experimental_semivar - theoretical_semivar
                rmse = np.sqrt((experimental_semivar - theoretical_semivar) ** 2)
                records.append([lag, experimental_semivar, theoretical_semivar, bias, rmse])

            pretty_table.add_rows(records)

            msg = header + pretty_table.get_string()
            return msg

    def __repr__(self):
        cname = 'TheoreticalVariogram'
        input_params = f'empirical_variogram={self.empirical_variogram}'
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

        lags = self.empirical_variogram.lags
        fitted_values = model_fn(lags, nugget, sill, rang)
        modeled = np.zeros(shape=(len(lags), 2))
        modeled[:, 0] = lags
        modeled[:, 1] = fitted_values
        return modeled

    def _set_model_parameters(self, model_params: dict):
        self.nugget = model_params['nugget']
        self.rang = model_params['range']
        self.sill = model_params['sill']
        self.name = model_params['name']


def build_theoretical_variogram(empirical_variogram: ExperimentalVariogram,
                                model_type: str,
                                sill: float,
                                rang: float,
                                nugget: float = 0.) -> TheoreticalVariogram:
    """Function is a wrapper into TheoreticalVariogram class and its fit() method.

    Parameters
    ----------
    empirical_variogram : ExperimentalVariogram

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
           Value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
           to observations variance.

    rang : float
           Semivariogram Range is a distance at which spatial correlation exists and often it is a distance when
           variogram reaches sill. It shouldn't be set at a distance larger than a half of a study extent.

    nugget : float, default=0.
             Nugget parameter (bias at a zero distance).

    Returns
    -------
    : TheoreticalVariogram

    """
    theo = TheoreticalVariogram()
    theo.fit(
        empirical_variogram=empirical_variogram, model_type=model_type, sill=sill, rang=rang, nugget=nugget
    )
    return theo
