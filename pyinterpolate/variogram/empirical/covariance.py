"""
The experimental covariance.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.processing.select_values import select_points_within_ellipse, select_values_in_range
from pyinterpolate.variogram.utils.exceptions import validate_direction, validate_points, validate_tolerance


def _create_empty_output(lag: float) -> list:
    """Function returns empty output for the case where no neighbors are selected.

    Parameters
    ----------
    lag : float

    Returns
    -------
    : list
        [lag, 0, 0]
    """
    return [lag, 0, 0]


# Covariogram calculations
def omnidirectional_covariogram(points: np.array, lags: np.array, step_size: float) -> np.array:
    """Function calculates covariance from given points.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values [pt x, pt y, value] or [Point(), value].

    lags : numpy array
           Specific lags (h) to calculate semivariance at a given range.

    step_size : float
                Distance between lags within a point pair is included in the calculation.

    Returns
    -------
    : numpy array
        [lag, covariance, number of points within lag]

    Notes
    -----
    covariance = 1 / (N) * SUM(i=1, N) [z(x_i + h) * z(x_i)] - u^2

    where:
        - N         - number of observation pairs,
        - h         - distance (lag),
        - z(x_i)    - value at location z_i,
        - (x_i + h) - location at a distance h from x_i,
        - u -         mean of observations at a given lag distance.
    """

    covariances_and_lags = list()
    distances = calc_point_to_point_distance(points[:, :-1])

    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        if len(distances_in_range[0]) == 0:
            if h == lags[0]:
                output = _create_empty_output(h)
                covariances_and_lags.append(output)
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
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
    """Function calculates directional semivariogram from a given set of points.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values [pt x, pt y, value].

    lags : numpy array
        List with specific lags (h) to calculate semivariance at a given range.

    step_size : float
                Distance between lags within a point pair is included in the calculation.

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
                    * Tolerance == 1 creates the omnidirectional covariogram.

    Returns
    -------
    : numpy array
        [lag, covariance, number of points within lag]
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
            if h == lags[0]:
                output = _create_empty_output(h)
                covariances_and_lags.append(output)
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
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
                         direction=None,
                         tolerance=1,
                         get_c0=True) -> tuple:
    """Function calculates covariance from given points. In a default mode it calculates an omnidirectional
       covariance. User can calculate a directional covariogram with a specified tolerance.

    Parameters
    ----------
    points : numpy array
             Coordinates and their values [pt x, pt y, value].

    step_size : float
                Distance between lags within a point pair is included in the calculation.

    max_range : float
                Maximum range of analysis. Lags are calculated from it as a points in range (0, max_range, step_size).

    direction : float, default=None
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
                    * Tolerance == 1 creates the omnidirectional covariogram.

    get_c0 : bool, default=True
             Calculate variance of a dataset and return it.

    Returns
    -------
    : tuple
        (numpy array [lag, covariance, number of pairs], variance : float or None)

    Notes
    -----
    # Covariance

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

    # Directional Covariogram

    Assumption that our observations change in the same way in every direction is rarely true. Let's consider
        temperature. It changes from equator to poles, so in the N-S and S-N axes. The good idea is to test if
        our observations are correlated in a few different directions. The main difference between an omnidirectional
        covariogram and a directional covariogram is that we take into account a different subset of neighbors.
        The selection depends on the direction (direction) of analysis. You may imagine it like that:

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
    >>> covariances = calculate_covariance(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(covariances[0][0])
    [ 1.         -0.54340278 24.        ]
    >>> print(covariances[1])
    4.2485207100591715
    """

    # START:VALIDATION
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
        covariance = omnidirectional_covariogram(points, lags, step_size)
    else:
        covariance = directional_covariogram(points, lags, step_size, direction, tolerance)
    # END:CALCULATIONS

    cvar = None

    if get_c0:
        cvar = np.var(points[:, -1])

    return covariance, cvar
