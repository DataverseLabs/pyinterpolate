"""
Tests for a data range.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""


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

    TODO
    ----
    Tests
    """

    msg = f'Value {value} is outside the limits {lower_limit}:{upper_limit}. Lower limit is excluded: ' \
          f'{exclusive_lower}, Upper limit is excluded" {exclusive_upper}.'

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
