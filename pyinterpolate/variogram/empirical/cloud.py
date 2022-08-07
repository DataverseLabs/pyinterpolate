import numpy as np
from collections import OrderedDict
from shapely.geometry import Point

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.processing.select_values import select_points_within_ellipse, select_values_in_range
from pyinterpolate.variogram.utils.exceptions import validate_direction, validate_points, validate_tolerance


def omnidirectional_point_cloud(input_array: np.array,
                                step_size: float,
                                max_range: float) -> dict:
    """
    Function calculates lagged omnidirectional point cloud.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    Returns
    -------
    variogram_cloud : dict
                      {Lag: array of semivariances within a given lag}
    """
    distances = calc_point_to_point_distance(input_array[:, :-1])
    lags = np.arange(step_size, max_range, step_size)
    variogram_cloud = OrderedDict()

    # Calculate squared differences
    # They are halved to be compatibile with semivariogram

    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        if len(distances_in_range[0]) == 0:
            variogram_cloud[h] = []
        else:
            sems = (input_array[distances_in_range[0], 2] - input_array[distances_in_range[1], 2]) ** 2
            variogram_cloud[h] = sems
    return variogram_cloud


def directional_point_cloud(input_array: np.array,
                            step_size: float,
                            max_range: float,
                            direction: float,
                            tolerance: float) -> dict:
    """
    Function calculates lagged variogram point cloud. Variogram is calculated as a squared difference of each point
        against other point within range specified by step_size parameter.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    direction : float (in range [0, 360])
                direction of semivariogram, values from 0 to 360 degrees:
                * 0 or 180: is NS direction,
                * 90 or 270 is EW direction,
                * 45 or 225 is NE-SW direction,
                * 135 or 315 is NW-SE direction.

    tolerance : float (in range [0, 1])
                If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                for 0 tolerance.
                * The minor axis size is (tolerance * step_size)
                * The major axis size is ((1 - tolerance) * step_size)
                * The baseline point is at a center of the ellipse.
                Tolerance == 1 creates an omnidirectional semivariogram.

    Returns
    -------
    variogram_cloud : dict
                      {Lag: array of semivariances within a given lag}
    """

    variogram_cloud = OrderedDict()
    lags = np.arange(step_size, max_range, step_size)

    for h in lags:
        variogram_vars_list = []
        for point in input_array:
            coordinates = point[:-1]

            mask = select_points_within_ellipse(
                coordinates,
                input_array[:, :-1],
                h,
                step_size,
                direction,
                tolerance
            )

            points_in_range = input_array[mask, -1]

            # Calculate semivariances
            if len(points_in_range) > 0:
                svars = (points_in_range - point[-1]) ** 2
                variogram_vars_list.extend(svars)

        if len(variogram_vars_list) == 0:
            variogram_cloud[h] = []
        else:
            variogram_cloud[h] = variogram_vars_list

    return variogram_cloud


def get_variogram_point_cloud(input_array: np.array,
                              step_size: float,
                              max_range: float,
                              direction=0,
                              tolerance=1) -> dict:
    """
    Function calculates lagged variogram point cloud. Variogram is calculated as a squared difference of each point
        against other point within range specified by step_size parameter.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value) or (Point(), value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    direction : float (in range [0, 360]), optional, default=0
                direction of semivariogram, values from 0 to 360 degrees:
                * 0 or 180: is NS direction,
                * 90 or 270 is EW direction,
                * 45 or 225 is NE-SW direction,
                * 135 or 315 is NW-SE direction.

    tolerance : float (in range [0, 1]), optional, default=1
                If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                for 0 tolerance.
                * The minor axis size is (tolerance * step_size)
                * The major axis size is ((1 - tolerance) * step_size)
                * The baseline point is at a center of the ellipse.
                Tolerance == 1 creates an omnidirectional semivariogram.

    Returns
    -------
    variogram_cloud : dict
                      {Lag: array of semivariances within a given lag}
    """

    # START:VALIDATION
    # Test size of points array and input data types
    validate_points(input_array)

    # Transform Point into floats
    is_point_type = isinstance(input_array[0][0], Point)
    if is_point_type:
        input_array = np.array([[x[0].x, x[0].y, x[1]] for x in input_array])

    # Test directions if provided
    validate_direction(direction)

    # Test provided tolerance parameter
    validate_tolerance(tolerance)
    # END:VALIDATION

    if tolerance == 1:
        return omnidirectional_point_cloud(input_array, step_size, max_range)
    else:
        return directional_point_cloud(input_array, step_size, max_range, direction, tolerance)
