from typing import Tuple

import numpy as np


def _weights_array(predicted_semivariances_shape, block_vals, point_support_vals) -> np.array:
    """
    Function calculates additional diagonal custom_weights for the matrix of predicted semivariances.

    Parameters
    ----------
    predicted_semivariances_shape : Tuple
        The size of semivariances array (nrows x ncols).

    block_vals : numpy array
        Array with values to calculate diagonal weight.

    point_support_vals : numpy array
        Array with values to calculate diagonal weight.

    Returns
    -------
    : numpy array
        The mask with zeros and diagonal weight of size (nrows x ncols).
    """

    weighted_array = np.sum(block_vals * point_support_vals)
    weight = weighted_array / np.sum(point_support_vals)
    w = np.zeros(shape=predicted_semivariances_shape)

    np.fill_diagonal(w, weight)
    return w
