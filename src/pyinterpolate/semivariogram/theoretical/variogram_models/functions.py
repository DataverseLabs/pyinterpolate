import numpy as np

from pyinterpolate.semivariogram.theoretical.variogram_models.models import \
    circular_model, cubic_model, exponential_model, gaussian_model, \
    linear_model, power_model, spherical_model

ALL_MODELS = ['circular',
              'cubic',
              'exponential',
              'gaussian',
              'linear',
              'power',
              'spherical']

SAFE_MODELS = ['linear', 'power', 'spherical']


class TheoreticalModelFunction:
    """Represents theoretical models

    Parameters
    ----------
    lags : numpy array
        Array of lags.

    nugget : float
        Semivariogram Nugget.

    sill : float
        Semivariogram Sill.

    rang : float
        Semivariogram Range.

    Attributes
    ----------
    lags : numpy array
        Lags.

    nugget : float
        Semivariogram Nugget.

    sill : float
        Semivariogram Sill.

    rang : float
        Semivariogram Range.

    models : list
        List of available models.

    yhat : numpy array
        Fitted values.

    Methods
    -------
    fit_predict()
        Fits a specific model to lags, nugget, sill, and range.
    """

    def __init__(self,
                 lags: np.ndarray,
                 nugget: float,
                 sill: float,
                 rang: float):
        self.lags = lags
        self.nugget = nugget
        self.sill = sill
        self.rang = rang
        self._model = {
            'circular': circular_model,
            'cubic': cubic_model,
            'exponential': exponential_model,
            'gaussian': gaussian_model,
            'linear': linear_model,
            'power': power_model,
            'spherical': spherical_model,

        }
        self.models = list(self._model.keys())
        self.yhat = None

    def fit_predict(self, model_type: str, return_values=True):
        """
        Method calculates semivariances.

        Parameters
        ----------
        model_type : str
            Model name from available models.

        return_values : bool, default = True
            Return fitted values.

        Returns
        -------
        : numpy array, optional
            Fitted values.
        """
        self._validate_model_name(model_type)

        mparams = {
            'lags': self.lags,
            'nugget': self.nugget,
            'sill': self.sill,
            'rang': self.rang
        }

        self.yhat = self._model[model_type](
            **mparams
        )
        if return_values:
            return self.yhat

    def _validate_model_name(self, model_type: str):
        """
        Checks if semivariogram model type is available.

        Parameters
        ----------
        model_type : str
            The name of the model to check.

        Raises
        ------
        KeyError
            Semivariogram model is not implemented.

        """
        if model_type not in self.models:
            msg = (f'Defined model name {model_type} is not implemented. '
                   f'You may choose one from {self.models} instead.')
            raise KeyError(msg)
