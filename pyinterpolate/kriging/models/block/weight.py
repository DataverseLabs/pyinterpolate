import numpy as np


def weights_array(predicted_semivariances_shape, block_vals, point_support_vals) -> np.array:
    """
    Function calculates additional diagonal weights for the matrix of predicted semivariances.

    Parameters
    ----------
    predicted_semivariances_shape : Tuple
                                    The size of semivariances array (nrows x ncols).

    block_vals : numpy array

    point_support_vals : numpy array

    Returns
    -------
    : numpy array
        The mask with zeros and diagonal weights.
    """

    weighted_array = np.sum(block_vals * point_support_vals)
    weight = weighted_array / np.sum(point_support_vals)
    w = np.ones(shape=predicted_semivariances_shape)
    np.fill_diagonal(w, weight)
    return w
