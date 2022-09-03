"""
Error metrics for the variogram model autofit.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
import numpy as np
from pyinterpolate.variogram.utils.exceptions import UndefinedSMAPEWarning


def forecast_bias(predicted_array: np.array, real_array: np.array) -> float:
    """Function calculates forecast bias of prediction.

    Parameters
    ----------
    predicted_array : numpy array
                      Array with predicted values.

    real_array : numpy array
                 Array with real observations.

    Returns
    -------
    fb : float
         Forecast Bias of prediction.

    Notes
    -----
    How do we interpret forecast bias? Here are two important properties:
        - Large positive value means that our observations are usually higher than prediction. Our model
          underestimates predictions.
        - Large negative value tells us that our predictions are usually higher than expected values. Our model
          overestimates predictions.

    Equation:

    (1) $$e_{fb} = \frac{\sum_{i}^{N}{y_{i} - \bar{y_{i}}}}{N}$$

        where:
        * $e_{fb}$ - forecast bias,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $N$ - number of observations.
    """

    fb = float(np.mean(real_array - predicted_array))
    return fb


def mean_absolute_error(predicted_array: np.array, real_array: np.array) -> float:
    """Function calculates forecast bias of prediction.

    Parameters
    ----------
    predicted_array : numpy array
                      Array with predicted values.

    real_array : numpy array
                 Array with real observations.

    Returns
    -------
    fb : float
         Forecast Bias of prediction.

    Notes
    -----
    MAE is not the same as RMSE. It is a good idea to compare the mean absolute error with the root mean squared error.
    We can get information about predictions that are very poor. RMSE that is larger than the MAE is a sign that for
    some lags our predictions are very poor. We should check those lags.

    Equation:

    (1) $$e_{mae} = \frac{\sum_{i}^{N}{|y_{i} - \bar{y_{i}}|}}{N}$$

        where:
        * $e_{mae}$ - mean absolute error,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $N$ - number of observations.
    """
    mae = float(
        np.mean(
            np.abs(real_array - predicted_array)
        )
    )
    return mae


def root_mean_squared_error(predicted_array: np.array, real_array: np.array) -> float:
    """

    Parameters
    ----------
    predicted_array : numpy array
                      Array with predicted values.

    real_array : numpy array
                 Array with real observations.

    Returns
    -------
    rmse : float
           Root Mean Squared Error of prediction.

    Notes
    -----
    Important hint: it is a good idea to compare the mean absolute error with the root mean squared error. We can get
    information about predictions that are very poor. RMSE that is larger than the MAE is a sign that for some
    lags our predictions are very poor. We should check those lags.

    Equation:

    (1) e_{rmse} = \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}}{N}}$$

        where:
        * $e_{rmse}$ - root mean squared error,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $N$ - number of observations.

    """
    rmse = np.sqrt(
        np.mean(
            (real_array - predicted_array)**2
        )
    )
    return rmse


def symmetric_mean_absolute_percentage_error(predicted_array: np.array, real_array: np.array,
                                             test_undefined=True) -> float:
    """

    Parameters
    ----------
    predicted_array : numpy array
                      Array with predicted values.

    real_array : numpy array
                 Array with real observations.

    test_undefined : bool
                     Check if there are cases when prediction and observation are equal to 0.

    Returns
    -------
    smape : float
            Symmetric Mean Absolute Percentage Error.

    Warns
    --------
    UndefinedSMAPEWarning
        Observation and prediction are equal to 0 - smape of this pair is undefined and numpy return the NaN value.

    Notes
    -----
    Symmetric Mean Absolute Percentage Error is an accuracy measure that returns error in percent. It is a relative
    evaluation metrics. It shouldn't be used alone because it can return different values for overforecast and
    underforecast. The SMAPE penalizes more underforecasting, thus it should be compared to Forecast Bias to have
    a full view of the model properties.
    SMAPE is better than RMSE or FB because it is better suited to compare multiple models with different number
    of parameteres, for example, number of ranges.

    More about SMAPE here: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Equation:

    (1) $$e_{smape} = \frac{100}{N} \sum_{i}^{N}{\frac{|\bar{y_{i}} - y_{i}|}{|y_{i}|+|\bar{y_{i}}|}}$$

        where:
        * $e_{smape}$ - symmetric mean absolute percentage error,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $N$ - number of observations.

    """
    smape = 100

    if test_undefined:
        psmapes = []
        for idx, val in enumerate(real_array):
            if val == 0 and predicted_array[idx] == 0:
                msg = f'Values of observation and prediction on position {idx} are equal to zero. ' \
                      f'Partial smape is set to 0 to avoid zero by zero division.'
                warnings.warn(msg, UndefinedSMAPEWarning)
                psmapes.append(0)
            else:
                pred = predicted_array[idx]
                psmape = np.abs(pred - val) / (np.abs(pred) + np.abs(val))
                psmapes.append(psmape)
        smape = smape * np.mean(psmapes)
    else:
        smape = smape * np.mean(
            np.abs(predicted_array - real_array) / (np.abs(real_array) + np.abs(predicted_array))
        )
    return smape


def weighted_root_mean_squared_error(predicted_array: np.array,
                                     real_array: np.array,
                                     weighting_method: str,
                                     lag_points_distribution=None) -> float:
    """

    Parameters
    ----------
    predicted_array : numpy array
                      Array with predicted values.

    real_array : numpy array
                 Array with real observations.

    weighting_method : str
                       The name of a method used to weight error at a given lags.
                       Available methods:
                       - closest: lags at a close range have larger weights,
                       - distant: lags that are further away have larger weights,
                       - dense: error is weighted by the number of point pairs within a lag.

    lag_points_distribution : numpy array
                              Array with points per lag (real observations).

    Returns
    -------
    wrmse : float
           Weighted Root Mean Squared Error of prediction.

    Raises
    ------
    AttributeError : The lag_points_distribution parameter is undefined when "dense" method is selected.

    Notes
    -----
    Error weighting is a useful method in the case when we want to force variogram to better represent semivariances
    at specific ranges. The most popular is "closest" - we create model that fits better pairs near the origin.

    Equations:

    - "closest"

    (1) $$e_{wrmse} = \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}*\frac{N-i}{N}}{N}}$$

        where:
        * $e_{rmse}$ - weighted root mean squared error,
        * $i$ - lag, i > 0,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $N$ - number of observations.

    - "distant"

    (2) $$e_{wrmse} = \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}*\frac{i}{N}}{N}}$$

        where:
        * $e_{rmse}$ - weighted root mean squared error,
        * $i$ - lag, i > 0,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $N$ - number of observations.

    - "dense"

    (3) $$e_{wrmse} = \sqrt{\frac{\sum_{i}^{N}({y_{i} - \bar{y_{i}})^2}*\frac{p_{i}}{P}}{N}}$$

        where:
        * $e_{rmse}$ - weighted root mean squared error,
        * $y_{i}$ - i-th observation,
        * $\bar{y_{i}}$ - i-th prediction,
        * $p_{i}$ - number of points within i-th lag,
        * $P$ - number of all points,
        * $N$ - number of observations.


    """
    error = (real_array - predicted_array)**2
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
            msg = 'The "dense" weighting method requires you to pass number of point pairs per lag!'
            raise AttributeError(msg)
        else:
            weights = lag_points_distribution / sum(lag_points_distribution)



    wrmse = error * weights
    wrmse = np.mean(wrmse)
    wrmse = np.sqrt(wrmse)

    return wrmse
