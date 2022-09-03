"""
Statistical functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Union, Iterable, Dict, List

import numpy as np
from scipy import stats


def detect_outliers_iqr(dataset: np.ndarray,
                        iqr_lower=1.5,
                        iqr_upper=1.5):
    """
    Function detects outliers in a dataset based on the inter-quartile ranges and standard deviation.

    Parameters
    ----------
    dataset : np.ndarray

    iqr_lower : float
        How many standard deviations from the 1st quartile down is the limit of expected values.

    iqr_upper : float
        How many standard deviations from the 3rd quartile up is the limit of expected values.

    Returns
    -------
    mask : np.ndarray
        Boolean mask with positions of outliers.

    Raises
    ------
    ValueError
        iqr_upper or iqr_lower are below zero.
    """

    if iqr_upper < 0 or iqr_lower < 0:
        msg = 'Parameters iqr_lower and iqr_upper must be floats greater or equal to 0.'
        raise ValueError(msg)

    lower_limit = np.quantile(dataset, q=0.25) - (np.std(dataset) * iqr_lower)
    upper_limit = np.quantile(dataset, q=0.75) + (np.std(dataset) * iqr_upper)

    mask = (dataset > upper_limit) | (dataset < lower_limit)
    return mask


def detect_outliers_z_score(dataset: np.ndarray,
                            z_lower=-3,
                            z_upper=3):
    """
    Function detects outliers in a given array.

    Parameters
    ----------
    dataset : np.ndarray

    z_lower : float
        How many standard deviations from the mean is an outlier (left tail).

    z_upper : float
        How many standard deviations from the mean is an outlier (left tail).

    Returns
    -------
    mask : np.ndarray
        Boolean mask with positions of outliers.

    Raises
    ------
    ValueError
        * z_dist_lower parameter is greater than 0.
        * z_dist_upper parameter is lower than 0.
        * z_dist_upper or z_dist_lower are equal to 0.
    """

    if z_lower >= 0:
        raise ValueError(f'The parameter z_lower must be a float lesser than zero.')
    if z_upper <= 0:
        raise ValueError(f'The parameter z_upper must be a float greater than zero.')

    outliers = stats.zscore(dataset)
    mask = (outliers > z_upper) | (outliers < z_lower)
    return mask


def remove_outliers(data: Union[Iterable, Dict],
                    method='zscore',
                    z_lower_limit=-3,
                    z_upper_limit=3,
                    iqr_lower_limit=1.5,
                    iqr_upper_limit=1.5) -> Union[List, Dict]:
    """
    Function removes outliers from a given dataset.

    Parameters
    ----------
    data : Union[Iterable, Dict]
        Dataset containing arrays (or lists) with a raw observations.

    method : str, default='zscore'
        Method used to detect outliers. Can be 'zscore' or 'iqr'.

    z_lower_limit : float
        Number of standard deviations from the mean to the left side of a distribution. Must be lower than 0.

    z_upper_limit : float
        Number of standard deviations from the mean to the right side of a distribution. Must be greater than 0.

    iqr_lower_limit : float
        Number of standard deviations from the 1st quartile into the lowest values. Must be greater or equal to zero.

    iqr_upper_limit : float
        Number of standard deviations from the 3rd quartile into the largest values. Must be greater or equal to zero.

    Returns
    -------
    cleaned : Union[List, Dict]
        List or Dict of transformed cleaned values.

    Raises
    ------
    KeyError
        Given detection method is not 'zscore' or 'iqr'.
    """

    detection_fns = {
        'zscore': [detect_outliers_z_score, z_lower_limit, z_upper_limit],
        'iqr': [detect_outliers_iqr, iqr_lower_limit, iqr_upper_limit]
    }

    # Check detection method
    if method not in detection_fns:
        msg = f'Given detection method: {method} is not available. Available methods are "zscore" and "iqr".'
        raise KeyError(msg)

    # Detect fn
    detect = detection_fns[method][0]
    _low = detection_fns[method][1]
    _high = detection_fns[method][2]

    if isinstance(data, Dict):
        cleaned = {}
        for _key, _values in data.items():
            if not isinstance(_values, np.ndarray):
                _values = np.array(_values)
            outliers_positions = detect(_values, _low, _high)
            ds = _values[~outliers_positions]
            cleaned[_key] = ds
    else:
        cleaned = []
        for _values in data:
            if not isinstance(_values, np.ndarray):
                _values = np.array(_values)
            outliers_positions = detect(_values, _low, _high)
            ds = _values[~outliers_positions]
            cleaned.append(ds)

    return cleaned
