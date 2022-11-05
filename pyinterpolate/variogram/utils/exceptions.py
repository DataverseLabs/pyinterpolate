"""
Additional exceptions and warnings for the variogram module.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings

import numpy as np


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


class VariogramModelNotSetError(Exception):
    """
    Exception raised when TheoreticalVariogram model name has not been set. (Model without name probably wasn't
    fitted).
    """

    def __init__(self):
        self.msg = 'Theoretical Variogram model is not set. You should fit() or autofit() TheoreticalVariogram ' \
                   'model first.'

    def __str__(self):
        return self.msg


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


class AttributeSetToFalseWarning(Warning):
    """
    Warning invoked when ExperimentalVariogram class attributes are set to False (is_semivariance, is_covariance,
    is_variance) but user wants to plot one of the indices controlled by those attributes (semivariance, covariance,
    variance) with a plot() method of this class.
    """
    def __init__(self, validated):
        wrong_params = list(validated.keys())
        msg = ''
        for _param in wrong_params:
            attr_msg = f'Warning! Attribute {_param} is set to False but you try to plot this object! Plot has been' \
                       f' cancelled.\n'
            msg = msg + attr_msg
        self.message = msg

    def __str__(self):
        return repr(self.message)


def validate_direction(direction):
    """
    Check if direction is within limits 0-360
    """
    if direction is not None:
        if np.abs(direction) > 360:
            msg = f'Provided direction must be between -360 to 360 degrees.'
            raise ValueError(msg)


def validate_points(points):
    """
    * Check dimensions of provided arrays and data types.
    * Check if there are any NaN values.
    """

    dims = points.shape
    msg = 'Provided array must have 3 columns: [x, y, value]'
    if dims[1] != 3:
        raise AttributeError(msg)

    if np.isnan(points[:, -1]).any():
        msg = 'Provided dataset contains NaNs, remove records with missing values before processing'
        raise ValueError(msg)


def validate_tolerance(tolerance):
    """
    Check if tolerance is between zero and one.
    """
    if tolerance <= 0 or tolerance > 1:
        msg = 'Provided tolerance should be larger than 0 (a straight line) and smaller or equal to 1 (a circle).'
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
    if minr <= 0:
        msg = f'Minimum range ratio is below 0 and it is equal to {minr}'
        raise ValueError(msg)

    # Check if max is larger than 1
    if maxr > 1:
        msg = f'Maximum range ratio should be lower than 1, but it is {maxr}'
        raise ValueError(msg)


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


def validate_plot_attributes_for_experimental_variogram_class(is_semivar: bool,
                                                              is_covar: bool,
                                                              is_var: bool,
                                                              plot_semivar: bool,
                                                              plot_covar: bool,
                                                              plot_var: bool):
    validation = {}

    if (is_semivar is False) and (plot_semivar is True):
        validation['is_semivariance'] = True

    if (is_covar is False) and (plot_covar is True):
        validation['is_covariance'] = True

    if (is_var is False) and (plot_var is True):
        validation['is_variance'] = True

    if validation:
        print(AttributeSetToFalseWarning(validation))


def validate_theoretical_variogram(variogram) -> None:
    """
    Function checks if variogram is set.

    Parameters
    ----------
    variogram : TheoreticalVariogram

    Returns
    -------
    : bool
        True if variogram is calculated.

    """
    if variogram.name is None:
        raise VariogramModelNotSetError()
