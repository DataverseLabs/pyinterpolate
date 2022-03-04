import numpy as np


def forecast_bias(predicted_array: np.array, real_array: np.array):
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
    """

    fb = np.mean(predicted_array - real_array)
    return fb
