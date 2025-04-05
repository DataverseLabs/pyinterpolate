import numpy as np


def parse_point_support_distances_array(distances: np.ndarray,
                                        values_a: np.ndarray,
                                        values_b: np.ndarray) -> np.ndarray:
    """
    Function parses given distances and values arrays into a single array.
    
    Parameters
    ----------
    distances : numpy array
        Array of size MxN with distances from each point from the set ``a``
        to each point from the set ``b``.

    values_a : numpy array
        Vector of length M with values. Represents values assigned
        to the points in the set ``a``.

    values_b : numpy array
        Vector of length N with values. Represents values assigned
        to the points in the set ``b``.

    Returns
    -------
    : numpy array
        ``[[value_a(i), value_b(j), distance(i-j)]]``
    """

    distances = distances.flatten()
    ldist = len(distances)

    alen = len(values_a)
    blen = len(values_b)
    mult_len = alen * blen

    if ldist != mult_len:
        raise AttributeError(
            "Distances length is different than the quotient of both "
            "points arrays length!"
        )

    a_values_arr = np.repeat(values_a, blen)
    b_values_arr = np.tile(values_b, alen)
    out_arr = np.array(list(zip(a_values_arr, b_values_arr, distances)))
    return out_arr


def add_ones(array: np.ndarray) -> np.ndarray:
    """
    Function adds rows of ones to a given array.

    Parameters
    ----------
    array : numpy array
        Array of size MxN (M rows, N cols)

    Returns
    -------
    list_with_ones : numpy array
        Array of size M+1xN (M+1 rows, N cols) where the last row are N ones.
    """
    ones = np.ones(np.shape(array)[1])
    list_with_ones = np.vstack((array, ones))
    return list_with_ones
