import numpy as np


def weight_experimental_semivariance(weights: np.ndarray,
                                     distances_in_range: np.ndarray,
                                     vals_0: np.ndarray,
                                     vals_h: np.ndarray) -> float:
    """
    Function calculates weighted semivariance.

    Parameters
    ----------
    weights : numpy array
        Weights of the semivariogram.

    distances_in_range : numpy array
        Distances between point pairs.

    vals_0 : numpy array
        Values of the first point in the pair.

    vals_h : numpy array
        Values of the second point in the pair.

    Returns
    -------
    : float
        Weighted semivariance.
    """

    # Weights
    ws_a = weights[distances_in_range[0]]
    ws_ah = weights[distances_in_range[1]]
    weights = (ws_a * ws_ah) / (ws_a + ws_ah)

    # sem
    weighted_vals_0 = vals_0 / ws_a
    weighted_vals_h = vals_h / ws_ah
    sem = (weighted_vals_0 - weighted_vals_h)**2

    # m'
    # bias term
    mweighted_mean = (np.mean(weighted_vals_0) + np.mean(weighted_vals_h)) / 2

    # numerator: w(0) * [(z(u_a) - z(u_a + h))^2] - m'
    numerator = weights * sem - mweighted_mean

    # semivariance
    sem_value = np.sum(numerator) / (2 * np.sum(weights))

    if sem_value < 0:
        return 0

    return sem_value
