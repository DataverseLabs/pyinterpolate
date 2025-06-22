from typing import Tuple

import numpy as np


def pk_weights_array(predicted_semivariances_shape: tuple,
                     block_vals: np.ndarray,
                     point_support_vals: np.ndarray) -> np.ndarray:
    """
    Function calculates additional diagonal custom weight in
    the matrix of predicted semivariances.

    Parameters
    ----------
    predicted_semivariances_shape : Tuple
        The size of semivariances array (nrows x ncols).

    block_vals : numpy array
        Aggregated values of blocks.

    point_support_vals : numpy array
        Total point support values in each block.

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
