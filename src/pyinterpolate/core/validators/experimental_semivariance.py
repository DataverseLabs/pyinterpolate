from pyinterpolate.core.validators.experimental_semivariance_warnings import \
    AttributeSetToFalseWarning


def validate_plot_attributes_for_experimental_variogram(
        is_semivar: bool,
        is_covar: bool,
        plot_semivar: bool,
        plot_covar: bool
):

    validation = {}

    semivar_test = (is_semivar is False) and (plot_semivar is True)
    covar_test = (is_covar is False) and (plot_covar is True)

    if semivar_test:
        validation['is_semivariance'] = True
    if covar_test:
        validation['is_covariance'] = True

    if validation:
        print(AttributeSetToFalseWarning(validation))


def validate_bins(step_size, max_range, custom_bins):
    """
    Function validates bins (lags) parameters.

    Parameters
    ----------
    step_size : float
    max_range : float
    custom_bins : Iterable
    """
    if step_size is None and max_range is None:
        if custom_bins is None:
            msg = 'You must provide at least `step_size` and `max_range` or ' \
                  '`custom_bins` parameter.'
            raise ValueError(msg)


def validate_direction_and_tolerance(direction, tolerance):
    """
    Function validates given direction and tolerance.

    Parameters
    ----------
    direction : float, optional
        Direction of semivariogram, values from -360 to 360 degrees.

    tolerance : float, optional
        Parameter that defines the shape of the bin, within limits (0:1].

    """
    if direction is not None and tolerance is not None:

        # Check if direction is in range 0-360
        if abs(direction) > 360:
            msg = 'Provided direction must be between -360 to 360 degrees.'
            raise ValueError(msg)

        # Check if tolerance is in range 0-1
        if tolerance <= 0 or tolerance > 1:
            msg = 'Provided tolerance should be larger than 0 ' \
                  '(a straight line) and smaller or equal to 1 (a circle).'
            raise ValueError(msg)
    else:
        if direction is None and tolerance is None:
            pass
        else:
            if direction is None and tolerance is not None:
                msg = 'Tolerance is set but direction is not set'
                raise ValueError(msg)
            else:
                msg = 'Direction is set but tolerance is not set'
                raise ValueError(msg)


def validate_semivariance_weights(points, weights):
    """
    Check if custom_weights array length is the same as points array.
    """
    if weights is not None:
        len_p = len(points)
        len_w = len(weights)
        _t = len_p == len_w
        # Check custom_weights and points
        if not _t:
            msg = f'Weights array length must be the same as length of points ' \
                  f'array but it has {len_w} records and' \
                  f' points array has {len_p} records'
            raise IndexError(msg)
        # Check if there is any 0 weight -> error
        if any([x == 0 for x in weights]):
            msg = 'One or more of custom_weights in dataset is set to 0, ' \
                  'remove point with a zero weight from a dataset, and ' \
                  'do not pass 0 as a weight.'
            raise ValueError(msg)
