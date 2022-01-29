import numpy as np
from shapely.geometry import Point
from pyinterpolate.processing.select_values import select_points_within_ellipse, select_values_in_range
from pyinterpolate.variogram.utils.validate import validate_direction, validate_points, validate_tolerance


# TEMPORARY FUNCTIONS
def temp_calc_point_to_point_distance(points_a, points_b=None):
    """temporary function for pt to pt distance estimation"""
    from scipy.spatial.distance import cdist
    if points_b is None:
        distances = cdist(points_a, points_a, 'euclidean')
    else:
        distances = cdist(points_a, points_b, 'euclidean')
    return distances


def _form_empty_output(lag_no) -> np.array:
    """
    Function returns empty output for the case where no neighbors are selected.

    :param lag_no: (int).

    :return: (numpy array)
    """
    return [lag_no, 0, 0]


# Covariogram calculations
def omnidirectional_covariogram(points: np.array, lags: np.array, step_size: float) -> np.array:
    """
    Function calculates covariance from given points.

    covariance = 1 / (N) * SUM(i=1, N) [z(x_i + h) * z(x_i)] - u^2

    where:
        - N         - number of observation pairs,
        - h         - distance (lag),
        - z(x_i)    - value at location z_i,
        - (x_i + h) - location at a distance h from x_i,
        - u -         mean of observations at a given lag distance.

    :param points: (numpy array) coordinates and their values:
        [pt x, pt y, value] or [Point(), value],
    :param lags: (numpy array) specific lags (h) to calculate semivariance at a given range,
    :param step_size: (float) distance between lags within each points are included in the calculations.

    :return: (numpy array) [lag, covariance, number of points within lag]
    """

    covariances_and_lags = list()
    distances = temp_calc_point_to_point_distance(points[:, :-1])

    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        if len(distances_in_range[0]) == 0:
            output = _form_empty_output(h)
            covariances_and_lags.append(output)
        else:
            vals_0 = points[distances_in_range[0], 2]
            vals_h = points[distances_in_range[1], 2]
            lag_mean = np.mean(vals_h)
            lag_mean_squared = lag_mean**2
            cov = (vals_0 * vals_h) - lag_mean_squared
            length = len(cov)
            cov_value = np.sum(cov) / length

            # Append calculated data into semivariance array
            covariances_and_lags.append([h, cov_value, length])

    output_covariances = np.vstack(covariances_and_lags)

    return output_covariances


def directional_covariogram(points: np.array,
                            lags: np.array,
                            step_size: float,
                            direction,
                            tolerance) -> np.array:
    """
    Function calculates directional semivariogram from a given set of points.

    :param points: (numpy array) coordinates and their values:
            [pt x, pt y, value]
    :param lags: (numpy array) specific lags (h) to calculate semivariance at a given range,
    :param step_size: (float) distance between lags within each points are included in the calculations,
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

    covariances_and_lags = list()

    for h in lags:
        covars_per_lag = []
        values_per_lag = []
        for point in points:

            coordinates = point[:-1]
            values_per_lag.append(point[-1])

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
                covars = (points_in_range * point[-1])
                values_per_lag.extend(points_in_range)
                covars_per_lag.extend(covars)

        if len(covars_per_lag) == 0:
            output = _form_empty_output(h)
            covariances_and_lags.append(output)
        else:
            lag_mean = np.mean(values_per_lag)
            lag_mean_squared = lag_mean * lag_mean
            average_covariance = np.mean(np.array(covars_per_lag) - lag_mean_squared)
            covariances_and_lags.append([h, average_covariance, len(covars_per_lag)])

    output_covariances = np.array(covariances_and_lags)

    return output_covariances


def calculate_covariance(points: np.array,
                         step_size: float,
                         max_range: float,
                         direction=0,
                         tolerance=1,
                         get_c0=True) -> tuple:
    """
    Function calculates covariance from given points. In a default mode it calculates an omnidirectional
        covariance. User can calculate directional covariogram with a specified tolerance.

    :param points: (numpy array) coordinates and their values:
            [pt x, pt y, value] or [Point(), value]
    :param step_size: (float) distance between lags within each points are included in the calculations,
    :param max_range: (float) maximum range of analysis,
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
        semivariogram,
    :param get_c0: (bool), default=True. Starts covariance array from the c(0) value which is a variance of a dataset.

    :return: (tuple) (np.array, float or None)

    ## Covariance

    It is a measure of similarity between points over distance. We assume that the close observations tends to be
        similar (see Tobler's Law). Distant observations are less and less similar up to the distance where influence
        of one point value into the other is negligible.

    We calculate the empirical covariance as:

        (1)    covariance = 1 / (N) * SUM(i=1, N) [z(x_i + h) * z(x_i)] - u^2

        where:

            - N         - number of observation pairs,
            - h         - distance (lag),
            - z(x_i)    - value at location z_i,
            - (x_i + h) - location at a distance h from x_i,
            - u -         mean of observations at a given lag distance.

    As an output we get array of lags h, covariances c and number of points within each lag n.

    ## Directional Covariogram

    Assumption that our observations change in the same way in every direction is rarely true. Let's consider
        temperature. It changes from equator to poles, so in the N-S and S-N axes. The good idea is to test if
        our observations are correlated in a few different directions. The main difference between an omnidirectional
        covariogram and a directional covariogram is that we take into account a different subset of neighbors.
        The selection depends on the angle (direction) of analysis. You may imagine it like that:

        - Omnidirectional covariogram: we test neighbors in a circle,
        - Directional covariogram: we test neighbors within an ellipse and one direction is major.

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
    # Test size of points array and input data types
    validate_points(points)

    # Transform Point into floats
    is_point_type = isinstance(points[0][0], Point)
    if is_point_type:
        points = [[x[0].x, x[0].y, x[1]] for x in points]

    # Test directions if provided
    validate_direction(direction)

    # Test provided tolerance parameter
    validate_tolerance(tolerance)
    # END:VALIDATION

    # START:CALCULATIONS
    lags = np.arange(step_size, max_range, step_size)

    if tolerance == 1:
        covariance = omnidirectional_covariogram(points, lags, step_size)
    else:
        covariance = directional_covariogram(points, lags, step_size, direction, tolerance)
    # END:CALCULATIONS

    cvar = None

    if get_c0:
        cvar = np.var(points[:, -1])

    return covariance, cvar
