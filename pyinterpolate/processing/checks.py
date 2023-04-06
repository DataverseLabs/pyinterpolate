"""
Tests for a data range.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
from typing import Collection
from pyinterpolate.processing.utils.exceptions import SetDifferenceWarning


def check_limits(value: float, lower_limit=0, upper_limit=1, exclusive_lower=True, exclusive_upper=True):
    """
    Function checks if value is within given limits. If not then ValueError is raised.

    Parameters
    ----------
    value : float

    lower_limit : float, default = 0

    upper_limit : float, default = 1

    exclusive_lower : bool, default = True
                      If set to True then value must be larger than the lower limit. If False then value must be equal
                      or larger than the lower limit.

    exclusive_upper : bool, default = True
                      If set to True then value must be smaller than the upper limit. If False then value must be
                      equal or smaller than the upper limit.

    Raises
    ------
    ValueError
        Value is outside given limits.
    """

    msg = f'Value {value} is outside the limits {lower_limit}:{upper_limit}. Lower limit is excluded: ' \
          f'{exclusive_lower}, Upper limit is excluded: {exclusive_upper}.'

    if value == lower_limit:
        if lower_limit == upper_limit:
            raise ValueError('Provided value, lower and upper limits are the same')

    # <=
    if exclusive_lower:
        if value <= lower_limit:
            raise ValueError(msg)

    # <
    if value < lower_limit:
        raise ValueError(msg)

    # >=
    if exclusive_upper:
        if value >= upper_limit:
            raise ValueError(msg)

    # >
    if value > upper_limit:
        raise ValueError(msg)


def check_ids(ids_a: Collection, ids_b: Collection, set_name_a='set 1', set_name_b='set 2'):
    """
    Function checks if there are any values that are missing in one set of ids, and present in another.

    Parameters
    ----------
    ids_a : Collection

    ids_b : Collection

    set_name_a : str, default='set 1'
        The name of the first set (for a warning message).

    set_name_b : str, default='set 2'
        The name of the second set (for a warning message).

    Warns
    -----

    """
    set_ids_a = set(ids_a)
    set_ids_b = set(ids_b)

    diff_a_to_b = set_ids_a.difference(set_ids_b)
    diff_b_to_a = set_ids_b.difference(set_ids_a)

    if len(diff_a_to_b) > 0 or len(diff_b_to_a) > 0:
        warnings.warn(SetDifferenceWarning(diff_a_to_b, diff_b_to_a, set_name_a, set_name_b))
