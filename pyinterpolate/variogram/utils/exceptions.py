import warnings
from shapely.geometry import Point


class MetricsTypeSelectionError(Exception):
    """Error invoked if user doesn't select any error type for the theoretical variogram modeling.

    Attributes
    ----------
    message : str
    """

    def __init__(self):
        self.message = "You didn't selected any error type from available rmse, bias, akaike and smape. Set one of" \
                       " those to True."

    def __str__(self):
        return self.message


class UndefinedSMAPEWarning(Warning):
    """Warning invoked by the scenario when predicted value is equal to 0 and observation is equal to 0. It leads to
        the 0/0 division and, in return, to NaN value at a specific position. Finally, user gets NaN as the output.

    Parameters
    ----------
    message : str

    Attributes
    ----------
    message : str
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def validate_direction(direction):
    """
    Check if direction is within limits 0-360
    """
    if direction < 0 or direction > 360:
        msg = f'Provided direction must be between 0 to 360 degrees:\n' \
              f'0-180-360: N-S\n' \
              f'90-270: E-W'
        raise ValueError(msg)


def validate_points(points):
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


def validate_tolerance(tolerance):
    """
    Check if tolerance is between zero and one.
    """
    if tolerance < 0 or tolerance > 1:
        msg = 'Provided tolerance should be between 0 (straight line) and 1 (circle).'
        raise ValueError(msg)


def validate_weights(points, weights):
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


def validate_selected_errors(val: int):
    if val == 0:
        raise MetricsTypeSelectionError


def check_ranges(minr: float, maxr: float):
    # Check if min is lower or equal to max
    if minr > maxr:
        msg = f'Minimum range {minr} is larger than maximum range {maxr}'
        raise ValueError(msg)

    # Check if min is negative
    if minr < 0:
        msg = f'Minimum range ratio is below 0 and it is equal to {minr}'
        raise ValueError(msg)

    # Check if max is larger than 1
    if maxr > 1:
        msg = f'Maximum range ratio should be lower than 1, but it is {maxr}'
        raise ValueError(msg)

    # Check if max is larger than 0.5 and throw warning if it is
    if maxr > 0.5:
        msg = f'Maximum range ratio is greater than the half of area smaller distance, it could introduce bias'
        warnings.warn(msg)


def check_sills(mins: float, maxs: float):
    # Check if min is lower or equal to max
    if mins > maxs:
        msg = f'Minimum sill ratio {mins} is larger than maximum sill ratio {maxs}'
        raise ValueError(msg)

    # Check if min is negative
    if mins < 0:
        msg = f'Minimum sill ratio is below 0 and it is equal to {mins}'
        raise ValueError(msg)

    # Check if max is larger than 0.5 and throw warning if it is
    if maxs > 1:
        msg = f'Maximum sill ratio is greater than the variance of a data, it could introduce bias'
        warnings.warn(msg)
