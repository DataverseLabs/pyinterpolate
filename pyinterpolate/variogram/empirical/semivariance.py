"""
Functions for calculating experimental semivariances.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import numpy as np
from pyinterpolate.processing.select_values import select_points_within_ellipse, select_values_in_range, \
    generate_triangles, select_points_within_triangle
from pyinterpolate.variogram.utils.exceptions import validate_direction, validate_points, validate_tolerance, \
    validate_weights

# Temp
from pyinterpolate.distance.distance import calc_point_to_point_distance


# Semivariogram calculations
def omnidirectional_semivariogram(points: np.array, lags: np.array, step_size: float, weights) -> np.array:
    """Function calculates semivariance from given points.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values: [pt x, pt y, value] or [Point(), value].

    lags : numpy array
           Array with lags.

    step_size : float
                Distance between lags.

    weights : numpy array
              Weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted by those.

    Returns
    -------
    : numpy array
        [lag, semivariance, number of points within a lag]
    """

    semivariances_and_lags = list()
    pts = points[:, :-1]
    distances = calc_point_to_point_distance(pts)

    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        if len(distances_in_range[0]) == 0:
            if h == lags[0]:
                semivariances_and_lags.append([h, 0, 0])
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
        else:

            vals_0 = points[distances_in_range[0], 2]
            vals_h = points[distances_in_range[1], 2]
            sem = (vals_0 - vals_h) ** 2
            length = len(sem)

            if weights is None:
                sem_value = np.sum(sem) / (2 * len(sem))
            else:
                # Weights
                ws_a = weights[distances_in_range[0]]
                ws_ah = weights[distances_in_range[1]]
                weights = (ws_a * ws_ah) / (ws_a + ws_ah)

                # m'
                mweighted_mean = np.average(vals_0,
                                            weights=ws_a)

                # numerator: w(0) * [(z(u_a) - z(u_a + h))^2] - m'
                numerator = weights * sem - mweighted_mean

                # semivariance
                sem_value = 0.5 * (np.sum(numerator) / np.sum(weights))

            # Append calculated data into semivariance array
            semivariances_and_lags.append([h, sem_value, length])

    output_semivariances = np.vstack(semivariances_and_lags)

    return output_semivariances


def _calculate_weighted_directional_semivariogram(points: np.array,
                                                  lags: np.array,
                                                  step_size: float,
                                                  weights: np.array,
                                                  direction: float,
                                                  tolerance: float) -> np.array:
    """Function calculates directional semivariogram from a given set of points.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values: [pt x, pt y, value] or [Point(), value].

    lags : numpy array
           Array with lags.

    step_size : float
                Distance between lags.

    weights : numpy array
              Weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted by those.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
                Value in range (0-1] to calculate semi-minor axis length of the search area. If tolerance is close
                to 0 then points must be placed at a single line with beginning in the origin of coordinate system
                and direction given by y axis and direction parameter.
                    * The major axis length == step_size,
                    * The minor axis size == tolerance * step_size.
                    * Tolerance == 1 creates the omnidirectional semivariogram.

    Returns
    -------
    : (numpy array)
        [lag, semivariance, number of points within a lag]
    """

    semivariances_and_lags = list()

    for h in lags:
        weighted_nominator_terms = []
        weighted_denominator_terms = []
        weighted_point_vals = []
        weight_of_points = []
        for idx, point in enumerate(points):
            coordinates = point[:-1]

            mask = select_points_within_ellipse(
                coordinates,
                points[:, :-1],
                h,
                step_size,
                direction,
                tolerance
            )

            points_in_range = points[mask, -1]

            # Calculate semivariances
            if len(points_in_range) > 0:
                w_vals = weights[mask]
                w_of_single_point = weights[idx]
                w_h = (w_of_single_point * w_vals) / (w_of_single_point + w_vals)
                z_h = (points_in_range - point[-1]) ** 2
                weighted_nominator_terms.extend(z_h)
                weighted_denominator_terms.extend(w_h)
                weighted_point_vals.extend(points_in_range)
                weight_of_points.extend(w_vals)

        if len(weighted_point_vals) == 0:
            if h == lags[0]:
                semivariances_and_lags.append([h, 0, 0])
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
            # TODO: remove after tests
            # semivariances_and_lags.append([h, 0, 0])
        else:
            arr_nom_terms = np.array(weighted_nominator_terms)
            arr_denom_terms = np.array(weighted_denominator_terms)
            arr_vals = np.array(weighted_point_vals)
            arr_weights = np.array(weight_of_points)

            nominator_z = arr_nom_terms - np.average(arr_vals, weights=arr_weights)
            nominator_z = np.sum(nominator_z * arr_denom_terms)
            gamma = nominator_z / np.sum(arr_denom_terms)
            average_semivariance = 0.5 * gamma
            semivariances_and_lags.append([h, average_semivariance, len(arr_nom_terms.flatten())])

    output_semivariances = np.array(semivariances_and_lags)
    return output_semivariances


def _from_ellipse_non_weighted(points: np.array, lags: np.array, step_size: float, direction, tolerance):
    """
    Function calculates semivariance from elliptical neighborhoods.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values: [pt x, pt y, value].

    lags : numpy array
           Array with lags.

    step_size : float
                Distance between lags.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
                Value in range (0-1] to calculate semi-minor axis length of the search area. If tolerance is close
                to 0 then points must be placed at a single line with beginning in the origin of coordinate system
                and direction given by y axis and direction parameter.
                    * The major axis length == step_size,
                    * The minor axis size == tolerance * step_size.
                    * Tolerance == 1 creates the omnidirectional semivariogram.

    Returns
    -------
    output_semivariances : numpy array
        ``[lag, semivariance, number of point pairs]``
    """
    semivariances_and_lags = list()

    for h in lags:
        semivars_per_lag = []

        for point in points:
            coordinates = point[:-1]

            mask = select_points_within_ellipse(
                coordinates,
                points[:, :-1],
                h,
                step_size,
                direction,
                tolerance
            )

            points_in_range = points[mask, -1]

            # Calculate semivariances
            if len(points_in_range) > 0:
                semivars = (points_in_range - point[-1]) ** 2
                semivars_per_lag.extend(semivars)

        if len(semivars_per_lag) == 0:
            if h == lags[0]:
                semivariances_and_lags.append([h, 0, 0])
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
        else:
            average_semivariance = np.mean(semivars_per_lag) / 2
            semivariances_and_lags.append([h, average_semivariance, len(semivars_per_lag)])

    output_semivariances = np.array(semivariances_and_lags)
    return output_semivariances


def create_triangles_mask(old_mask, new_mask):
    mask = []
    for idx, val in enumerate(new_mask):
        if old_mask[idx]:
            mask.append(False)
        else:
            if val:
                mask.append(True)
            else:
                mask.append(False)
    return mask


def _from_triangle_non_weighted(points: np.array, lags: np.array, direction, tolerance):
    """
    Function calculates semivariance from elliptical neighborhoods.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values: [pt x, pt y, value] or [Point(), value].

    lags : numpy array
           Array with lags.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
                Value in range (0-1] to calculate semi-minor axis length of the search area. If tolerance is close
                to 0 then points must be placed at a single line with beginning in the origin of coordinate system
                and direction given by y axis and direction parameter.
                    * The major axis length == step_size,
                    * The minor axis size == tolerance * step_size.
                    * Tolerance == 1 creates the omnidirectional semivariogram.

    Returns
    -------
    output_semivariances : numpy array
        ``[lag, semivariance, number of point pairs]``
    """
    semivariances_and_lags = list()

    old_mask = None

    for h in lags:
        semivars_per_lag = []
        trs = generate_triangles(points[:, :-1], h, direction, tolerance)
        if h == lags[0]:
            for idx, pt in enumerate(points):
                tr = trs[idx]
                points_in_triangle_a = select_points_within_triangle(tr[0], points[:, :-1])
                points_in_triangle_b = select_points_within_triangle(tr[1], points[:, :-1])
                mask = np.logical_or(points_in_triangle_a, points_in_triangle_b)
                # Update last mask
                old_mask = mask.copy()
                points_in_range = points[mask]
                values_in_range = points_in_range[:, -1]
                # Calculate semivariances
                if len(values_in_range) > 0:
                    semivars = (values_in_range - pt[-1]) ** 2
                    semivars_per_lag.extend(semivars)
            if len(semivars_per_lag) == 0:
                semivariances_and_lags.append([h, 0, 0])
            else:
                average_semivariance = np.mean(semivars_per_lag) / 2
                semivariances_and_lags.append([h, average_semivariance, len(semivars_per_lag)])
        else:
            for idx, pt in enumerate(points):
                tr = trs[idx]
                points_in_triangle_a = select_points_within_triangle(tr[0], points[:, :-1])
                points_in_triangle_b = select_points_within_triangle(tr[1], points[:, :-1])
                new_mask = np.logical_or(points_in_triangle_a, points_in_triangle_b)
                mask = create_triangles_mask(old_mask, new_mask)
                old_mask = new_mask.copy()
                points_in_range = points[mask]
                values_in_range = points_in_range[:, -1]
                if len(values_in_range) > 0:
                    semivars = (values_in_range - pt[-1]) ** 2
                    semivars_per_lag.extend(semivars)
            if len(semivars_per_lag) == 0:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
            else:
                average_semivariance = np.mean(semivars_per_lag) / 2
                semivariances_and_lags.append([h, average_semivariance, len(semivars_per_lag)])

    output_semivariances = np.array(semivariances_and_lags)
    return output_semivariances


def directional_semivariogram(points: np.array,
                              lags: np.array,
                              step_size: float,
                              weights,
                              direction,
                              tolerance,
                              method='triangle') -> np.array:
    """Function calculates directional semivariogram from a given set of points.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values: [pt x, pt y, value] or [Point(), value].

    lags : numpy array
           Array with lags.

    step_size : float
                Distance between lags.

    weights : numpy array
              Weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted by those.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
                Value in range (0-1] to calculate semi-minor axis length of the search area. If tolerance is close
                to 0 then points must be placed at a single line with beginning in the origin of coordinate system
                and direction given by y axis and direction parameter.
                    * The major axis length == step_size,
                    * The minor axis size == tolerance * step_size.
                    * Tolerance == 1 creates the omnidirectional semivariogram.

    method : str, default = triangular
        The method used for neighbors selection. Available methods:

        * "triangle" or "t", default method where a point neighbors are selected from a triangular area,
        * "ellipse" or "e", the most accurate method but also the slowest one.

    Returns
    -------
    : (numpy array)
      [lag, semivariance, number of points within a lag]
    """

    output_semivariances = np.array([])

    if weights is None:
        if method == "e" or method == "ellipse":
            output_semivariances = _from_ellipse_non_weighted(points, lags, step_size, direction, tolerance)
        elif method == "t" or method == "triangle":
            output_semivariances = _from_triangle_non_weighted(points, lags, direction, tolerance)
    else:
        output_semivariances = _calculate_weighted_directional_semivariogram(
            points, lags, step_size, weights, direction, tolerance
        )

    return output_semivariances


def calculate_semivariance(points: np.array,
                           step_size: float,
                           max_range: float,
                           weights=None,
                           direction=None,
                           tolerance=1,
                           method='t') -> np.array:
    """Function calculates semivariance from given points. In a default mode it calculates non-weighted and
       omnidirectional semivariance. User may provide weights to additionally weight points by a specific factor.
       User can calculate directional semivariogram with a specified tolerance.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values: [pt x, pt y, value] or [Point(), value].

    step_size : float
                Distance between lags.

    max_range : float
                Maximum range of analysis. Lags are calculated from it as a points in range (0, max_range, step_size).

    weights : numpy array or None, optional, default=None
              Weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted by those.

    direction : float, default = None
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
                Value in range (0-1] to calculate semi-minor axis length of the search area. If tolerance is close
                to 0 then points must be placed at a single line with beginning in the origin of coordinate system
                and direction given by y axis and direction parameter.
                    * The major axis length == step_size,
                    * The minor axis size == tolerance * step_size.
                    * Tolerance == 1 creates the omnidirectional semivariogram.

    method : str, default = triangular
        The method used for neighbors selection. Available methods:

        * "triangle" or "t", default method where a point neighbors are selected from a triangular area,
        * "ellipse" or "e", the most accurate method but also the slowest one.

    Returns
    -------
    : (numpy array)
        [lag, semivariance, number of points within a lag]

    Notes
    -----
    # Semivariance

    It is a measure of dissimilarity between points over distance. We assume that the close observations tends to be
        similar (see Tobler's Law). Distant observations are less and less similar up to the distance where influence
        of one point value into the other is negligible.

    We calculate the empirical semivariance as:

        (1)    g(h) = 0.5 * n(h)^(-1) * (SUM|i=1, n(h)|: [z(x_i + h) - z(x_i)]^2)

        where:
            h: lag,
            g(h): empirical semivariance for lag h,
            n(h): number of point pairs within a specific lag,
            z(x_i): point a (value of observation at point a),
            z(x_i + h): point b in distance h from point a (value of observation at point b).

    As an output we get array of lags h, semivariances g and number of points within each lag n.

    # Weighted Semivariance

    For some cases we need to weight each point by specific factor. It is especially important for the semivariogram
        deconvolution and Poisson Kriging. We can weight observations by a specific factors, for example the time effort
        for observation at a location (ecology) or population size at a specific block (public health).
        Implementation of algorithm follows publications:

        1. A. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C: Comparison of model based geostatistical methods
        in ecology: application to fin whale spatial distribution in northwestern Mediterranean Sea.
        In Geostatistics Banff 2004 Volume 2. Edited by: Leuangthong O, Deutsch CV. Dordrecht, The Netherlands,
        Kluwer Academic Publishers; 2005:777-786.
        2. B. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C: Geostatistical modelling of spatial distribution
        of Balenoptera physalus in the northwestern Mediterranean Sea from sparse count data and heterogeneous
        observation efforts. Ecological Modelling 2006 in press.

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
            z(x_i + h): point b in distance h from point a (rate of specific process at point b),
            n(x_i): denominator value size at point a (time, population ...),
            n(x_i + h): denominator value size at point b in distance h from point a,
            m': weighted mean of rates.

    The output of weighted algorithm is the same as for non-weighted data: array of lags h, semivariances g and number
        of points within each lag n.

    # Directional Semivariogram

    Assumption that our observations change in the same way in every direction is rarely true. Let's consider
        temperature. It changes from equator to poles, so in the N-S and S-N axes. The good idea is to test if
        our observations are correlated in a few different directions. The main difference between an omnidirectional
        semivariogram and a directional semivariogram is that we take into account a different subset of neighbors.
        The selection depends on the direction (direction) of analysis. You may imagine it like that:

        - Omnidirectional semivariogram: we test neighbors in a circle,
        - Directional semivariogram: we test neighbors within an ellipse and one direction is major.

        or graphically:

        - omnidirectional:

               ooooo
              oooxooo
               ooooo

        - directional (W-E):

              oooxooo

        - directional with some tolerance (N-S):

                 o
                 o
                oxo
                 o
                 o

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
    >>> semivariances = calculate_semivariance(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(semivariances[0])
    [ 1.     4.625 24.   ]
    """

    # START:VALIDATION
    if weights is not None:
        validate_weights(points, weights)

    # Test size of points array and input data types
    validate_points(points)

    # Test directions if provided
    validate_direction(direction)

    # Test provided tolerance parameter
    validate_tolerance(tolerance)
    # END:VALIDATION

    # START:CALCULATIONS
    lags = np.arange(step_size, max_range, step_size)

    if direction is None:
        semivariance = omnidirectional_semivariogram(points, lags, step_size, weights)
    else:
        semivariance = directional_semivariogram(points, lags, step_size, weights, direction, tolerance, method=method)
    # END:CALCULATIONS
    return semivariance
