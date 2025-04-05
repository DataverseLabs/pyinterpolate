from typing import Union, List

import numpy as np


def get_lags(step_size, max_range, custom_bins):
    """
    Function creates array of lags.

    Parameters
    ----------
    step_size : float
        Step size between lags.

    max_range : float
        Maximum range.

    custom_bins : Union[Iterable, np.ndarray]
        Custom lags given by user.

    Returns
    -------
    lags : np.ndarray
    """
    if custom_bins is not None:
        if custom_bins[0] == 0:
            return np.array(custom_bins)[1:]
        return np.array(custom_bins)

    else:
        return np.arange(step_size, max_range, step_size)


def get_current_and_previous_lag(lag_idx: int, lags: Union[List, np.ndarray]):
    """
    Function returns current and previous lag positions.

    Parameters
    ----------
    lag_idx : int
        Index of the current lag.

    lags : Iterable
        Array of lags.

    Returns
    -------
    : float, float
        current lag, previous lag
    """
    if lag_idx == 0:
        current_lag = lags[lag_idx]
        previous_lag = 0
    else:
        current_lag = lags[lag_idx]
        previous_lag = lags[lag_idx - 1]

    return current_lag, previous_lag
