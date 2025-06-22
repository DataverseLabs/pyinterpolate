"""
Error metrics for the variogram model autofit.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
import numpy as np


class UndefinedSMAPEWarning(Warning):
    """
    Warning invoked by the scenario when predicted value is equal to 0
    and observation is equal to 0. It leads to the 0/0 division and,
    in return, to the ``NaN`` value at a specific position.
    Finally, user gets ``NaN`` as the output.

    Parameters
    ----------
    zero_idx : int
        Index where observation and prediction are equal to 0.

    Attributes
    ----------
    message : str
        Warning message.
    """

    def __init__(self, zero_idx: int):
        msg = (f'Values of observation and prediction on position {zero_idx} '
               f'are equal to zero. Partial smape is set to 0 to avoid zero '
               f'by zero division.')
        self.message = msg

    def __str__(self):
        return repr(self.message)


def forecast_bias(predicted_array: np.ndarray,
                  real_array: np.ndarray) -> float:
    r"""
    Function calculates forecast bias of prediction.

    Parameters
    ----------
    predicted_array : numpy array
        Predicted values.

    real_array : numpy array
        Real observations.

    Returns
    -------
    fb : float
        Forecast Bias of prediction.

    Notes
    -----
    How do we interpret forecast bias? Here are two important properties:
        - Large positive value means that our observations are usually higher
          than prediction. Our model underestimates predictions.
        - Large negative value tells us that our predictions are usually
          higher than expected values. Our model overestimates predictions.

    Equation:

    .. math:: e_{fb} = \frac{\sum_{i}^{N}{y_{i} - \bar{y_{i}}}}{N}

    where:
        - :math:`e_{fb}` - forecast bias,
        - :math:`y_{i}` - i-th observation,
        - :math:`\bar{y_{i}}` - i-th prediction,
        - :math:`N` - number of observations.
    """

    fb = float(np.mean(real_array - predicted_array))
    return fb


def mean_absolute_error(predicted_array: np.ndarray,
                        real_array: np.ndarray) -> float:
    r"""
    Function calculates Mean Absolute Error (MAE) of prediction.

    Parameters
    ----------
    predicted_array : numpy array
        Predicted values.

    real_array : numpy array
        Observations.

    Returns
    -------
    mae : float
        Mean absolute Error (MAE) of prediction.

    Equation:

    .. math:: e_{mae} = \frac{\sum_{i}^{N}{|y_{i} - \bar{y_{i}}|}}{N}

    where:
        * :math:`e_{mae}` - mean absolute error,
        * :math:`y_{i}` - i-th observation,
        * :math:`\bar{y_{i}}` - i-th prediction,
        * :math:`N` - number of observations.
    """
    mae = float(
        np.mean(
            np.abs(real_array - predicted_array)
        )
    )
    return mae


def root_mean_squared_error(predicted_array: np.ndarray,
                            real_array: np.ndarray) -> float:
    r"""
    Function calculates Root Mean Squared Error of predictions.

    Parameters
    ----------
    predicted_array : numpy array
        Predictions.

    real_array : numpy array
        Observations.

    Returns
    -------
    rmse : float
        Root Mean Squared Error of prediction.

    Notes
    -----
    Equation:

    .. math:: e_{rmse} = \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}}{N}}

    where:
        * :math:`e_{rmse}` - root mean squared error,
        * :math:`y_{i}` - i-th observation,
        * :math:`\bar{y_{i}}` - i-th prediction,
        * :math:`N` - number of observations.

    """
    rmse = np.sqrt(
        np.mean(
            (real_array - predicted_array) ** 2
        )
    )
    return rmse


def symmetric_mean_absolute_percentage_error(predicted_array: np.ndarray,
                                             real_array: np.ndarray,
                                             test_undefined=True) -> float:
    r"""
    Function calculates Symmetric Mean Absolute Percentage Error (SMAPE) of
    predictions, allowing researcher to compare different models.

    Parameters
    ----------
    predicted_array : numpy array
        Predictions.

    real_array : numpy array
        Observations

    test_undefined : bool, default = True
        Check if there are cases when prediction and observation are
        equal to 0.

    Returns
    -------
    smape : float
            Symmetric Mean Absolute Percentage Error.

    Warns
    -----
    UndefinedSMAPEWarning
        Observation and prediction are equal to 0 - SMAPE of this pair is
        undefined, algorithm assumes that SMAPE equals to 0.

    Notes
    -----
    Symmetric Mean Absolute Percentage Error is an accuracy measure that
    returns prediction error in percent. It is a relative evaluation metric.
    It shouldn't be used alone because SMAPE penalizes more underforecasting,
    thus it should be compared to Forecast Bias to have a full view of the
    model properties. SMAPE is better than RMSE or FB for comparing multiple
    models and algorithms.

    More about SMAPE here:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Equation:

    .. math:: e_{smape} = \frac{100}{N}
        \sum_{i}^{N}{\frac{|\bar{y_{i}} - y_{i}|}{|y_{i}|+|\bar{y_{i}}|}}

    where:
        * :math:`e_{smape}` - symmetric mean absolute percentage error,
        * :math:`y_{i}` - i-th observation,
        * :math:`\bar{y_{i}}` - i-th prediction,
        * :math:`N` - number of observations.

    """
    smape = 100

    if test_undefined:
        psmapes = []
        for idx, val in enumerate(real_array):
            if val == 0 and predicted_array[idx] == 0:
                warnings.warn(idx, UndefinedSMAPEWarning)
                psmapes.append(0)
            else:
                pred = predicted_array[idx]
                psmape = np.abs(pred - val) / (np.abs(pred) + np.abs(val))
                psmapes.append(psmape)
        smape = smape * np.mean(psmapes)
    else:
        smape = smape * np.mean(
            np.abs(predicted_array - real_array) /
            (np.abs(real_array) + np.abs(predicted_array))
        )
    return smape


def weighted_root_mean_squared_error(predicted_array: np.ndarray,
                                     real_array: np.ndarray,
                                     weighting_method: str,
                                     lag_points_distribution=None) -> float:
    r"""
    Function weights RMSE of each lag by a specific weighting factor.

    Parameters
    ----------
    predicted_array : numpy array
        Predictions.

    real_array : numpy array
        Observations.

    weighting_method : str
        The name of a method used to weight error at
        a given lags.
        Available methods:
        - closest: lags at a close range have greater weights,
        - distant: lags that are further away have greater weights,
        - dense: error is weighted by the number of point pairs within a lag.

    lag_points_distribution : numpy array, optional
        Number of points pairs per lag.

    Returns
    -------
    wrmse : float
        Weighted Root Mean Squared Error.

    Raises
    ------
    AttributeError :
        The ``lag_points_distribution`` parameter is undefined when
        "dense" method is set.

    Notes
    -----
    Error weighting is a useful in the case when we want to
    force semivariogram to better represent semivariances at specific ranges.
    The most popular is the ``"closest"`` method - we create model that
    fits better semivariogram at a close distances.

    Equations:

    ``"closest"``

    .. math:: e_{wrmse} =
        \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}*\frac{N-i}{N}}{N}}

    where:
        * :math:`e_{rmse}` - weighted root mean squared error,
        * :math:`i` - lag, i > 0,
        * :math:`y_{i}` - i-th observation,
        * :math:`\bar{y_{i}}` - i-th prediction,
        * :math:`N` - number of observations.

    ``"distant"``

    .. math:: e_{wrmse} =
        \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}*\frac{i}{N}}{N}}

    where:
        * :math:`e_{rmse}` - weighted root mean squared error,
        * :math:`i` - lag, i > 0,
        * :math:`y_{i}` - i-th observation,
        * :math:`\bar{y_{i}}` - i-th prediction,
        * :math:`N` - number of observations.

    ``"dense"``

    .. math:: e_{wrmse} =
        \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}*\frac{p_{i}}{P}}{N}}

    where:
        * :math:`e_{rmse}` - weighted root mean squared error,
        * :math:`y_{i}` - i-th observation,
        * :math:`\bar{y_{i}}` - i-th prediction,
        * :math:`p_{i}` - number of points within i-th lag,
        * :math:`P` - number of all points,
        * :math:`N` - number of observations.

    """
    error = (real_array - predicted_array) ** 2
    arr_length = len(error)
    weights = np.ones(len(error))

    if weighting_method == 'closest':
        sequence = np.arange(1, arr_length + 1)
        weights = (arr_length - sequence) / arr_length
    elif weighting_method == 'distant':
        sequence = np.arange(1, arr_length + 1)
        weights = sequence / arr_length
    elif weighting_method == 'dense':
        if lag_points_distribution is None:
            msg = ('The "dense" weighting method requires '
                   'passing the number of point pairs per lag!')
            raise AttributeError(msg)
        else:
            weights = lag_points_distribution / sum(lag_points_distribution)

    wrmse = error * weights
    wrmse = np.mean(wrmse)
    wrmse = np.sqrt(wrmse)

    return wrmse
