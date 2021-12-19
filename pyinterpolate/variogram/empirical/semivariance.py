import numpy as np
from shapely.geometry import Point
from pyinterpolate.processing.select_values import select_points_within_ellipse, select_values_in_range


# TEMPORARY FUNCTIONS
def temp_calc_point_to_point_distance(points_a, points_b=None):
    """temporary function for pt to pt distance estimation"""
    from scipy.spatial.distance import cdist
    if points_b is None:
        distances = cdist(points_a, points_a, 'euclidean')
    else:
        distances = cdist(points_a, points_b, 'euclidean')
    return distances


# TESTS AND EXCEPTIONS
def _validate_direction(direction):
    """
    Check if direction is within limits 0-360
    """
    if direction < 0 or direction > 360:
        msg = f'Provided direction must be between 0 to 360 degrees:\n' \
              f'0-180-360: N-S\n' \
              f'90-270: E-W'
        raise ValueError(msg)


def _validate_points(points):
    """
    Check dimensions of provided arrays and data types.
    """

    dims = points.shape
    msg = 'Provided array must have 3 columns: [x, y, value] or 2 columns: [shapely Point(), value]'
    if dims[1] != 3:
        if dims[1] == 2:
            # Check if the first value is a Point type
            if not isinstance(points[0][0], Point):
                raise AttributeError(msg)
        else:
            raise AttributeError(msg)


def _validate_tolerance(tolerance):
    """
    Check if tolerance is between zero and one.
    """
    if tolerance < 0 or tolerance > 1:
        msg = 'Provided tolerance should be between 0 (straight line) and 1 (circle).'
        raise ValueError(msg)


def _validate_weights(points, weights):
    """
    Check if weights array length is the same as points array.
    """
    len_p = len(points)
    len_w = len(weights)
    _t = len_p == len_w
    # Check weights and points
    if not _t:
        msg = f'Weights array length must be the same as length of points array but it has {len_w} records and' \
              f' points array has {len_p} records'
        raise IndexError(msg)
    # Check if there is any 0 weight -> error
    if any([x == 0 for x in weights]):
        msg = 'One or more of weights in dataset is set to 0, this may cause errors in the distance'
        raise ValueError(msg)


# Semivariogram calculations
def _omnidirectional_semivariogram(points: np.array, lags: np.array, step_size: float, weights) -> np.array:
    """
    Function calculates semivariance from given points.

    :param points: (numpy array) coordinates and their values:
        [pt x, pt y, value] or [Point(), value],
    :param lags: (numpy array) specific lags (h) to calculate semivariance at a given range,
    :param step_size: (float) distance between lags within each points are included in the calculations,
    :param weights: (numpy array) weights assigned to points, index of weight must be the same as index of point, if
        provided then semivariogram is weighted by those.


    :return: (numpy array) [lag, semivariance, number of points within lag]
    """

    semivariances_and_lags = list()
    distances = temp_calc_point_to_point_distance(points[:, :-1])

    for h in lags:
        # TODO: temp func
        distances_in_range = select_values_in_range(distances, h, step_size)
        if len(distances_in_range[0]) == 0:
            semivariances_and_lags.append([h, 0, 0])
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


def _directional_semivariogram(points: np.array,
                               lags: np.array,
                               step_size: float,
                               weights,
                               direction,
                               tolerance) -> np.array:
    """
    Function calculates directional semivariogram from a given set of points.

    :param points: (numpy array) coordinates and their values:
            [pt x, pt y, value] or [Point(), value]
    :param lags: (numpy array) specific lags (h) to calculate semivariance at a given range,
    :param step_size: (float) distance between lags within each points are included in the calculations,
    :param weights: (numpy array) weights assigned to points, index of weight must be the same as index of point, if
        provided then semivariogram is weighted by those,
    :param direction: (float) direction of semivariogram, values from 0 to 360 degrees:
        0 or 180: is NS direction,
        90 or 270 is EW direction,
        45 or 225 is NE-SW direction,
        135 or 315 is NW-SE direction,
    :param tolerance: (float) value in range (0-1) normalized to [0 : 0.5] to select tolerance of semivariogram.
        If tolerance is 0 then points must be placed at a single line with beginning in the origin of coordinate system
        and angle given by y axis and direction parameter. If tolerance is greater than 0 then semivariance is estimated
        from elliptical area with major axis with the same direction as the line for 0 tolerance and minor axis
        of a size:
        (tolerance * step_size)
        and major axis (pointed in NS direction):
        ((1 - tolerance) * step_size)
        and baseline point at a center of ellipse. Tolerance == 1 (normalized to 0.5) creates omnidirectional
        semivariogram.

    :returns: (np.array)
    """

    semivariances_and_lags = list()

    for h in lags:
        semivars_per_lag = []
        for point in points:
            mask = select_points_within_ellipse(
                point,
                points,
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
            else:
                semivars_per_lag.append(0)

        average_semivariance = np.mean(semivars_per_lag) / 2
        semivariances_and_lags.append([h, average_semivariance, len(semivars_per_lag)])

    output_semivariances = np.array(semivariances_and_lags)
    return output_semivariances


def calculate_semivariance(points: np.array,
                           step_size: float,
                           max_range: float,
                           weights=None,
                           direction=0,
                           tolerance=1) -> np.array:
    """
    Function calculates semivariance from given points. In a default mode it calculates non-weighted and omnidirectional
        semivariance. User may provide weights to additionally weight points by a specific factor. User can calculate
        directional semivariogram with a specified tolerance.

    :param points: (numpy array) coordinates and their values:
            [pt x, pt y, value] or [Point(), value]
    :param step_size: (float) distance between lags within each points are included in the calculations,
    :param max_range: (float) maximum range of analysis,
    :param weights: (numpy array) weights assigned to points, index of weight must be the same as index of point, if
        provided then semivariogram is weighted by those,
    :param direction: (float) direction of semivariogram, values from 0 to 360 degrees:
        0 or 180: is NS direction,
        90 or 270 is EW direction,
        45 or 225 is NE-SW direction,
        135 or 315 is NW-SE direction,
    :param tolerance: (float) value in range (0-1) normalized to [0 : 0.5] to select tolerance of semivariogram.
        If tolerance is 0 then points must be placed at a single line with beginning in the origin of coordinate system
        and angle given by y axis and direction parameter. If tolerance is greater than 0 then semivariance is estimated
        from elliptical area with major axis with the same direction as the line for 0 tolerance and minor axis
        of a size:
        (tolerance * step_size)
        and major axis (pointed in NS direction):
        ((1 - tolerance) * step_size)
        and baseline point at a center of ellipse. Tolerance == 1 (normalized to 0.5) creates omnidirectional
        semivariogram.

    ## Semivariance

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

    ## Weighted Semivariance

    For some cases we need to weight each point by specific factor. It is especially important for the semivariogram
        deconvolution and Poisson Kriging. We can weight observations by specific factors, for example the time effort
        for observations at specific locations (ecology) or population size at specific block (public health).
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

    ## Directional Semivariogram

    Assumption that our observations change in the same way in every direction is rarely true. Let's consider
        temperature. It changes from equator to poles, so in the N-S and S-N axes. The good idea is to test if
        our observations are correlated in a few different directions. The main difference between an omnidirectional
        semivariogram and a directional semivariogram is that we take into account a different subset of neighbors.
        The selection depends on the angle (direction) of analysis. You may imagine it like that:

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

    ## Examples:

    # TODO


    """

    # START:VALIDATION
    # Data validity tests
    if weights is not None:
        _validate_weights(points, weights)

    # Test size of points array and input data types
    _validate_points(points)

    # Transform Point into floats
    is_point_type = isinstance(points[0][0], Point)
    if is_point_type:
        points = [[x[0].x, x[0].y, x[1]] for x in points]

    # Test directions if provided
    _validate_direction(direction)

    # Test provided tolerance parameter
    _validate_tolerance(tolerance)
    # END:VALIDATION

    # START:CALCULATIONS

    lags = np.arange(step_size, max_range, step_size)

    if tolerance == 1:
        semivariance = _omnidirectional_semivariogram(points, lags, step_size, weights)
    else:
        semivariance = _directional_semivariogram(points, lags, step_size, weights, direction, tolerance)

    # END:CALCULATIONS

    return semivariance
