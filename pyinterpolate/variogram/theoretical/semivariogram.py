import numpy as np
from collections import namedtuple
from pyinterpolate.variogram.empirical import EmpiricalVariogram
from pyinterpolate.variogram.utils.evals import forecast_bias
from pyinterpolate.variogram.utils.errors import check_selected_errors


class TheoreticalVariogram:
    """Theoretical model of spatial data dissimilarity.

    Parameters
    ----------
    empirical_variogram : EmpiricalVariogram
                          Prepared Empirical Variogram.

    verbose : bool, default=False
              Prints messages related to the model preparation.

    Attributes
    ----------
    verbose : bool, default=False
              Prints messages related to the model preparation.

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

    model : str or None, default=None
            The name of a theoretical model. Only finite set of models are available.
            See variogram_models attribute.

    name : str or None, default=None
           Name of the chosen model. Available names are the same as keys in variogram_models attribute.

    nugget : float, default=0
             Nugget parameter (bias at a zero distance).

    sill : float, default=0
           Value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
           to observations variance.

    range : float, default=0
            Range is a distance at which spatial correlation exists and often it is a distance when variogram reaches
            sill. It shouldn't be set at a distance larger than a half of a study extent.

    rmse : float, default=0
           Root mean squared error of the difference between the empirical observations and the modeled curve.

    bias : float, default=0
           Forecast Bias of the estimation. Large positive value means that the estimated model usually overestimates
           values and large negative value means that model underestimates predictions.

    smape : float, default=0
            Symmetric Mean Absolute Percentage Error of the prediction - values from 0 to 100%.

    akaike : float, default=0
             Akaike information criterion (AIC) of a given model. Quality of a model.

    """

    def __init__(self,
                 empirical_variogram: EmpiricalVariogram,
                 verbose=False):

        self.verbose = verbose

        # Model
        self.empirical_variogram = empirical_variogram
        self.variogram_models = {
            'circular': self._circular_model,
            'cubic': self._cubic_model,
            'exponential': self.exponential_model,
            'gaussian': self._gaussian_model,
            'linear': self._linear_model,
            'power': self._power_model,
            'spherical': self._spherical_model
        }

        # Model parameters
        self.model = None
        self.name = None
        self.nugget = 0.
        self.range = 0.
        self.sill = 0.

        # Dynamic parameters
        self.rmse = 0.
        self.bias = 0.
        self.smape = 0.
        self.akaike = 0.

    def fit(self,
            model_type: str,
            sill: float,
            range: float,
            nugget=0.,
            update_attrs=True) -> namedtuple:
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

        range : float
                Range is a distance at which spatial correlation exists and often it is a distance when variogram
                reaches its sill. It shouldn't be set at a distance larger than a half of a study extent.

        nugget : float, default=0.
                 Nugget parameter (bias at a zero distance),

        update_attrs : bool, default=True
                       Should class attributes be updated?

        Raises
        ------
        KeyError : Model type not implemented

        Returns
        -------
        ModelErrors : namedtuple
                      ('ModelErrors', 'rmse bias smape akaike')

        """

        # Check model type

        _names = list(self.variogram_models.keys())

        if not model_type in _names:
            msg = f'Defined model name {model_type} not available. You may choose one from {names} instead.'
            raise KeyError(msg)
        
        _model = self.variogram_models[model_type]

        _theoretical_values = _model(nugget, sill, range)
        
        # Estimate errors
        _error = self.calculate_model_error(_model, nugget, sill, range,
                                            rmse=True, bias=True, akaike=True, smape=True)
        return _error

    def autofit(self,
                model_type=None,
                nugget=None,
                min_range=0,
                max_range=0.5,
                number_of_ranges=16,
                min_sill=0.,
                max_sill=1,
                number_of_sills=16,
                error_estimator='rmse'):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    #TODO: annot to _model
    def calculate_model_error(self, model_fn,
                              nugget: float,
                              sill: float,
                              range: float,
                              rmse=True,
                              bias=True,
                              akaike=True,
                              smape=True) -> namedtuple:
        """

        Parameters
        ----------
        model_fn
        nugget
        sill
        range
        rmse
        bias
        akaike
        smape

        Returns
        -------

        Raises
        ------
        ErrorTypeSelectionError : User set all errors to False
        """

        # Check errors
        check_selected_errors(rmse + bias + akaike + smape)
        ModelErrors = namedtuple('ModelErrors', 'rmse bias akaike smape')

        _model_values = model_fn(self.empirical_variogram.experimental_semivariance_array[:, 0], nugget, sill, range)
        _real_values = self.empirical_variogram.experimental_semivariance_array[:, 1].copy()

        # Get Forecast Biast
        if bias:
            _fb = forecast_bias(_model_values, _real_values)
        else:
            _fb = np.nan

        # Get RMSE
        if rmse:
            _rmse = root_mean_squared_error(_model_values, _real_values)
        else:
            _rmse = np.nan

        if akaike:
            _akaike = akaike_information_criterion(nugget, sill, range, len(_real_values))
        else:
            _akaike = np.nan

        if smape:
            _smape = symmetric_mean_absolute_percentage_error(_model_values, _real_values)
        else:
            _smape = np.nan

        model_errors = ModelErrors(_rmse, _fb, _akaike, _smape)
        return model_errors