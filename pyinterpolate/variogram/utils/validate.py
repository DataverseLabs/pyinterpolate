from shapely.geometry import Point


# TESTS AND EXCEPTIONS
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
