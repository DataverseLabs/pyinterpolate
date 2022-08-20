import numpy as np
from scipy import stats


def detect_outliers_z_score(dataset: np.ndarray,
                            z_dist_lower=-3,
                            z_dist_upper=3):
    """
    Function detects outliers in a given array.

    Parameters
    ----------
    dataset : np.ndarray

    z_dist_lower : float
        How many standard deviations from the mean is an outlier (left tail).

    z_dist_upper : float
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

    if z_dist_lower >= 0:
        raise ValueError(f'The parameter z_dist_lower must be a float lesser than zero.')
    if z_dist_upper <= 0:
        raise ValueError(f'The parameter z_dist_upper must be a float greater than zero.')

    outliers = stats.zscore(dataset)
    mask = outliers[outliers > z_dist_upper] or outliers[outliers < z_dist_lower]
    return mask


def remove_outliers(data: Union[Iterable, Dict],
                    method='zscore',
                    zdist_lower_limit=-3,
                    zdist_upper_limit=3,
                    iqr_lower_limit=1.5,
                    iqr_upper_limit=1.5):
    pass
