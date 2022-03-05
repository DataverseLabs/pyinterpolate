import numpy as np


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


def symmetric_mean_absolute_percentage_error(predicted_array: np.array, real_array: np.array) -> float:
    """

    Parameters
    ----------
    predicted_array : numpy array
                      Array with predicted values.

    real_array : numpy array
                 Array with real observations.

    Returns
    -------
    smape : float
            Symmetric Mean Absolute Percentage Error.

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
    smape = 100 * np.mean(
        np.abs(predicted_array - real_array) / (np.abs(real_array) + np.abs(predicted_array))
    )
    return smape
