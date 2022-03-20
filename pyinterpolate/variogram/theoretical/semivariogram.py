from typing import Collection, Union, Callable, Tuple

import numpy as np
from collections import namedtuple

from prettytable import PrettyTable

from pyinterpolate.processing.select_values import create_min_max_array
from pyinterpolate.variogram.theoretical.models import circular_model, cubic_model, linear_model, exponential_model, \
    gaussian_model, spherical_model, power_model
from pyinterpolate.variogram.empirical import EmpiricalVariogram
from pyinterpolate.variogram.utils.evals import forecast_bias, root_mean_squared_error,\
    symmetric_mean_absolute_percentage_error, mean_absolute_error
from pyinterpolate.variogram.utils.exceptions import validate_selected_errors, check_ranges, check_sills


class TheoreticalVariogram:
    """Theoretical model of spatial data dissimilarity.

    Parameters
    ----------
    empirical_variogram : EmpiricalVariogram
                          Prepared Empirical Variogram.

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
    """

    def __init__(self, empirical_variogram: EmpiricalVariogram):

        # Model
        self.empirical_variogram = empirical_variogram
        self.variogram_models = {
            'circular': circular_model,
            'cubic': cubic_model,
            'exponential': exponential_model,
            'gaussian': gaussian_model,
            'linear': linear_model,
            'power': power_model,
            'spherical': spherical_model
        }

        # Model parameters
        self.fitted_model = None
        self.name = None
        self.nugget = 0.
        self.rang = 0.
        self.sill = 0.

        # Dynamic parameters
        self.rmse = 0.
        self.bias = 0.
        self.smape = 0.
        self.mae = 0.

    def fit(self,
            model_type: str,
            sill: float,
            rang: float,
            nugget=0.,
            update_attrs=True) -> Tuple[np.array, namedtuple]:
        """

        Parameters
        ----------
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

        Raises
        ------
        KeyError : Model type not implemented

        Returns
        -------
        : Tuple[ numpy array, namedtuple ]
            [ theoretical semivariances, ('ModelError', 'rmse bias smape mae')]

        """

        # Check model type

        _names = list(self.variogram_models.keys())

        if model_type not in _names:
            msg = f'Defined model name {model_type} not available. You may choose one from {_names} instead.'
            raise KeyError(msg)
        
        _model = self.variogram_models[model_type]

        _theoretical_values = self._fit_model(_model, nugget, sill, rang)
        
        # Estimate errors
        _error = self.calculate_model_error(_model, rmse=True, bias=True, smape=True)

        if update_attrs:
            self._update_attributes(
                fitted_model=_theoretical_values,
                model_type=model_type,
                nugget=nugget,
                sill=sill,
                rang=rang,
                bias=_error.bias,
                smape=_error.smape,
                rmse=_error.rmse
            )

        return _theoretical_values, _error

    def autofit(self,
                model_types: Union[str, list],
                nugget=0,
                min_range=0,
                max_range=0.5,
                number_of_ranges=16,
                min_sill=0.,
                max_sill=1,
                number_of_sills=16,
                error_estimator='rmse',
                auto_update_attributes=True,
                verbose=False):

        """Methods tries to find the optimal range, sill and model of theoretical semivariogram.

        Parameters
        ----------
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

        min_range : float, default = 0
                    Minimal fraction of a variogram range, 0 <= min_range <= max_range

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

        Raises
        ------
        ValueError : sill < 0 or range < 0 or range > 1.

        KeyError : wrong model name(s) or wrong error type name.
        """

        # Check parameters
        check_ranges(min_range, max_range)
        check_sills(min_sill, max_sill)

        # Check model type and set models
        mtypes = self._check_models_type_autofit(model_types)

        # Set ranges and sills
        dist_range = self.empirical_variogram.lags[-1]
        min_max_ranges = create_min_max_array(dist_range, min_range, max_range, number_of_ranges)

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
            'rang': 0,
            'error_type': error_estimator,
            'error_value': err_val
        }

        for _mtype in mtypes:
            for _rang in min_max_ranges:
                for _sill in min_max_sill:
                    # Create model
                    _mdl_fn = self.variogram_models[_mtype]
                    _fitted_model = self._fit_model(_mdl_fn, nugget, _sill, _rang)
                    _err = self.calculate_model_error(_fitted_model, **_errors_keys)

                    if verbose:
                        self.__print_autofit_info(_mtype, nugget, _sill, _rang, error_estimator, _err[error_estimator])

                    # Check if model is better than the previous
                    if _err[error_estimator] < err_val:
                        err_val = _err[error_estimator]
                        optimal_parameters['model_type'] = _mtype
                        optimal_parameters['nugget'] = nugget
                        optimal_parameters['sill'] = _sill
                        optimal_parameters['rang'] = _rang
                        optimal_parameters['error_type'] = error_estimator
                        optimal_parameters['error_value'] = err_val
                        optimal_parameters['fitted_model'] = _mdl_fn
        
        if auto_update_attributes:
            self._update_attributes(**optimal_parameters)

        return optimal_parameters

    def __str__(self):
        pretty_table = PrettyTable()
        # TODO add title to pretty table
        title = f'{self.name}'.capitalize()
        header = title + '\n'
        pretty_table.field_names = ["lag", "experimental", "theoretical", "bias", "rmse"]

        pass

    def __repr__(self):
        pass

    def plot(self, experimental=True):
        pass

    def calculate_model_error(self,
                              fitted_values: np.array,
                              rmse=True,
                              bias=True,
                              mae=True,
                              smape=True) -> namedtuple:
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
        model_errors : namedtuple
                       Named tuple with error values per model: rmse, bias, mae, smape.

        Raises
        ------
        MetricsTypeSelectionError : User set all errors to False
        """

        # Check errors
        validate_selected_errors(rmse + bias + mae + smape)  # all False sums to 0 -> error detection
        ModelError = namedtuple('ModelError', 'rmse bias mae smape')

        _real_values = self.empirical_variogram.experimental_semivariance_array[:, 1].copy()

        # Get Forecast Biast
        if bias:
            _fb = forecast_bias(fitted_values, _real_values)
        else:
            _fb = np.nan

        # Get RMSE
        if rmse:
            _rmse = root_mean_squared_error(fitted_values, _real_values)
        else:
            _rmse = np.nan

        if mae:
            _mae = mean_absolute_error(fitted_values, _real_values)
        else:
            _mae = np.nan

        if smape:
            _smape = symmetric_mean_absolute_percentage_error(fitted_values, _real_values)
        else:
            _smape = np.nan

        model_errors = ModelError(_rmse, _fb, _mae, _smape)
        return model_errors

    def _check_model_names(self, mname):
        _names = list(self.variogram_models.keys())

        if mname not in _names:
            msg = f'Defined model name {mname} not available. You may choose one from {_names} instead.'
            raise KeyError(msg)

    def _check_models_type_autofit(self, model_types: Union[str, Collection]):
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
        modeled = np.concatenate([lags, fitted_values], axis=1)
        return modeled


def build_theoretical_variogram(empirical_variogram: EmpiricalVariogram,
                                model_type: str,
                                sill: float,
                                rang: float,
                                nugget: float = 0.) -> TheoreticalVariogram:
    """Function is a wrapper into TheoreticalVariogram class and its fit() method.

    Parameters
    ----------
    empirical_variogram : EmpiricalVariogram

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
    theo = TheoreticalVariogram(empirical_variogram)
    theo.fit(
        model_type=model_type, sill=sill, rang=rang, nugget=nugget
    )
    return theo