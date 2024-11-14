from typing import Union, List

import numpy as np


def get_lags(step_size, max_range, custom_bins):
    """
    Function creates array of lags.

    Parameters
    ----------
    step_size : float
    max_range : float
    custom_bins : Union[Iterable, np.ndarray]

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
    Function returns y_exp and previous lag.

    Parameters
    ----------
    lag_idx : int

    lags : Iterable

    Returns
    -------
    : float, float
        y_exp lag, previous lag
    """
    if lag_idx == 0:
        current_lag = lags[lag_idx]
        previous_lag = 0
    else:
        current_lag = lags[lag_idx]
        previous_lag = lags[lag_idx - 1]

    return current_lag, previous_lag
