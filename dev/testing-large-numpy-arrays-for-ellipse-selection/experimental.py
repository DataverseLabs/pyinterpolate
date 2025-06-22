import threading
from typing import Union, List, Collection

import numpy as np

from pyinterpolate.core.data_models.points import RawPoints
from pyinterpolate.core.validators.experimental_semivariance import \
    validate_semivariance_weights, validate_direction_and_tolerance, \
    validate_bins
from pyinterpolate.distance.point import point_distance,\
    select_values_in_range
from pyinterpolate.semivariogram.experimental.functions.directional import \
    from_ellipse_non_weighted
from pyinterpolate.semivariogram.lags.lags import get_lags
from pyinterpolate.semivariogram.weights.experimental.weighting import \
    weight_experimental_semivariance


def calculate_semivariance(ds: np.ndarray,
                           step_size: float = None,
                           max_range: float = None,
                           direction: float = None,
                           tolerance: float = None,
                           dir_neighbors_selection_method: str = 't',
                           custom_bins: Union[np.ndarray, Collection] = None,
                           custom_weights: np.ndarray = None) -> np.ndarray:
    """
    Calculates experimental semivariance.

    Parameters
    ----------
    ds : numpy array
        ``[x, y, value]``

    step_size : float
        The fixed distance between lags grouping point neighbors.

    max_range : float
        The maximum distance at which the semivariance is calculated.

    direction : float, optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    dir_neighbors_selection_method : str, default = 't'
        The dir_neighbors_selection_method used for neighbors selection. Available methods:

        * "triangle" or "t", default dir_neighbors_selection_method where a point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", the most accurate dir_neighbors_selection_method but also the slowest one.

    custom_bins : numpy array, optional
        Custom bins for semivariance calculation. If provided, then parameter
        ``step_size`` is ignored and ``max_range`` is set to the final bin
        distance.

    custom_weights : numpy array, optional
        Custom weights assigned to points.

    Returns
    -------
    semivariance : numpy array
        ``[lag, semivariance, number of point pairs]``

    Notes
    -----
    # Semivariance

    It is a measure of dissimilarity between points over distance.
    We assume that the close observations tend to be similar (see Tobler's
    Law). Distant observations are less and less similar up to the distance
    where the influence of one point value on the other is negligible.


    We calculate the empirical semivariance as:

        (1)    g(h) = 0.5 * n(h)^(-1) * (
                      SUM|i=1, n(h)|: [z(x_i + h) - z(x_i)]^2
                      )

        where:
            h: lag,
            g(h): empirical semivariance for lag h,
            n(h): number of point pairs within a specific lag,
            z(x_i): point a (value of observation at point a),
            z(x_i + h): point b in distance h from point a (value of
              observation at point b).

    As an output we get array of lags h, semivariances g and number of
    points within each lag n.

    # Weighted Semivariance

    Sometimes, we need to weight each point by a specific factor.
    It is especially important for the semivariogram deconvolution and
    Poisson Kriging. The weighting factor could be the time effort for
    observation at a location (ecology) or population size at a specific
    block (public health). Implementation of the algorithm follows
    publications:


        1. A. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C:
        Comparison of model based geostatistical methods in ecology:
        application to fin whale spatial distribution in northwestern
        Mediterranean Sea. In Geostatistics Banff 2004 Volume 2.
        Edited by: Leuangthong O, Deutsch CV. Dordrecht, The Netherlands,
        Kluwer Academic Publishers; 2005:777-786.
        2. B. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C:
        Geostatistical modelling of spatial distribution of Balenoptera
        physalus in the northwestern Mediterranean Sea from sparse count data
        and heterogeneous observation efforts. Ecological Modelling 2006.

    We calculate the weighted empirical semivariance as:

        (2)    g_w(h) = 0.5 * (SUM|i=1, n(h)|: w(h))^(-1) * ...
                            * (SUM|i=1, n(h)|: w(h) * z_w(h))

        (3)    w(h) = [n(x_i) * n(x_i + h)] / [n(u_i) + n(u_i + h)]

        (4)    z_w(h) = (z(x_i) - z(x_i + h))^2 - m'

        where:
            h: lag,
            g_w(h): weighted empirical semivariance for lag h,
            n(h): number of point pairs within a specific lag,
            z(x_i): point a (rate of specific process at point a),
            z(x_i + h): point b in distance h from point a (rate of specific
              process at point b),
            n(x_i): denominator value size at point a (time, population ...),
            n(x_i + h): denominator value size at point b in distance h from
              point a,
            m': weighted mean of rates.

    The output of weighted algorithm is the same as for non-weighted data:
    array of lags h, semivariances g and number of points within each lag n.

    # Directional Semivariogram

    The assumption that our observations change in the same way in every
    direction is rarely true. Let's consider temperature. It changes from
    the equator to the poles so in the N-S and S-N axes. The good idea is to
    test the correlation of our observations in a few different directions.
    The main difference between an omnidirectional semivariogram and
    a directional semivariogram is that we take into account a different
    subset of neighbors:

       - Omnidirectional semivariogram: we test neighbors in a circle,
       - Directional semivariogram: we test neighbors within an ellipse,
         and one direction is major.

    Examples
    --------
    >>> import numpy as np
    >>> REFERENCE_INPUT = np.array([
    ...    [0, 0, 8],
    ...    [1, 0, 6],
    ...    [2, 0, 4],
    ...    [3, 0, 3],
    ...    [4, 0, 6],
    ...    [5, 0, 5],
    ...    [6, 0, 7],
    ...    [7, 0, 2],
    ...    [8, 0, 8],
    ...    [9, 0, 9],
    ...    [10, 0, 5],
    ...    [11, 0, 6],
    ...    [12, 0, 3]
    ...    ])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> semivariances = calculate_semivariance(
    ...    REFERENCE_INPUT,
    ...    step_size=STEP_SIZE,
    ...    max_range=MAX_RANGE)
    >>> print(semivariances[0])
    [ 1.     4.625 24.   ]
    """

    # Validation
    # Validate points
    ds = RawPoints(points=ds)

    # Validate bins
    validate_bins(step_size, max_range, custom_bins)

    # Validate custom_weights
    validate_semivariance_weights(ds.points, custom_weights)

    # Validate direction and tolerance
    validate_direction_and_tolerance(direction, tolerance)

    # Calculations
    # Get lags
    lags = get_lags(step_size, max_range, custom_bins)

    # Get semivariances
    if direction is not None and tolerance is not None:
        experimental_semivariances = directional_semivariance(
            ds.points,
            lags,
            direction,
            tolerance,
            dir_neighbors_selection_method,
            custom_weights
        )
    else:
        experimental_semivariances = omnidirectional_semivariance(
            ds.points, lags, custom_weights
        )

    return experimental_semivariances


def directional_semivariance(points: np.ndarray,
                             lags: Union[List, np.ndarray],
                             direction: float,
                             tolerance: float,
                             method: str,
                             custom_weights: np.ndarray = None):
    """
    Function calculates directional semivariances.

    Parameters
    ----------
    points : numpy array
             ``[x, y, value]``

    lags : numpy array

    direction : float, optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    method : str
        The dir_neighbors_selection_method used for neighbors selection. Available methods:

        * "triangle" or "t", default dir_neighbors_selection_method where a point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", the most accurate dir_neighbors_selection_method but also the slowest one.

    custom_weights : optional, Iterable
        Custom weights assigned to points.

    Returns
    -------
    : (numpy array)
      ``[lag, semivariance, number of point pairs]``
    """
    output_semivariances = np.array([])

    if custom_weights is None:
        if method == "e" or method == "ellipse":
            output_semivariances = from_ellipse_non_weighted(points,
                                                             lags,
                                                             direction,
                                                             tolerance)
        elif method == "t" or method == "triangle":
            output_semivariances = _from_triangle_non_weighted(points,
                                                               lags,
                                                               direction,
                                                               tolerance)
    else:
        output_semivariances = _calculate_weighted_directional_semivariogram(
            points, lags, step_size, weights, direction, tolerance
        )

    return output_semivariances



def omnidirectional_semivariance(points: np.ndarray,
                                 lags: Union[List, np.ndarray],
                                 custom_weights: np.ndarray):
    """
    Function calculates the omnidirectional semivariances.

    Parameters
    ----------
    points : numpy array

    lags : Iterable

    custom_weights : Iterable

    Returns
    -------
    semivariances : np.ndarray
    """
    coordinates = points[:, :-1]

    distances = point_distance(
        points=coordinates,
        other=coordinates
    )

    semivariances = []

    def _get(lag_idx):
        semivariances.append(
            _get_omnidirectional_semivariances_and_lags(
                points=points,
                distances=distances,
                lags=lags,
                lag_idx=lag_idx,
                weights=custom_weights
            )
        )

    threads = []
    for idx in range(len(lags)):
        thread = threading.Thread(
            target=_get,
            args=(idx, )
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return np.array(semivariances)


def _get_omnidirectional_semivariances_and_lags(
        points: np.ndarray,
        distances: np.ndarray,
        lags: Union[List, np.ndarray],
        lag_idx: int,
        weights: np.ndarray = None
):
    """
    Function calculates semivariance for a given lag, returning lag,
    semivariance and number of point pairs in range.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    distances : numpy array
        Distances from each point to other points.

    lags : list or numpy array
        The list of lags.

    lag_idx : int
        Current lag index.

    weights : numpy array, optional

    Returns
    -------
    : Tuple
        ``(lag, semivariance, number of point pairs within a lag)``

    """
    if lag_idx == 0:
        current_lag = lags[lag_idx]
        previous_lag = 0
    else:
        current_lag = lags[lag_idx]
        previous_lag = lags[lag_idx - 1]

    distances_in_range = select_values_in_range(distances,
                                                current_lag,
                                                previous_lag)

    if len(distances_in_range[0]) == 0:
        if lag_idx == 0:
            return current_lag, 0, 0
        else:
            msg = f'There are no neighbors for a lag {current_lag},' \
                  f'the process has been stopped.'
            raise RuntimeError(msg)
    else:
        vals_0 = points[distances_in_range[0], 2]
        vals_h = points[distances_in_range[1], 2]
        length = vals_0.size

        if weights is None:
            sem = np.square(vals_0 - vals_h)
            sem_value = np.sum(sem) / (2 * length)
        else:
            sem_value = weight_experimental_semivariance(
                weights=weights,
                distances_in_range=distances_in_range,
                vals_0=vals_0,
                vals_h=vals_h
            )

        return current_lag, sem_value, length
