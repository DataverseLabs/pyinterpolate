"""
General-purpose validators

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""


def check_limits(value: float,
                 lower_limit=0,
                 upper_limit=1,
                 exclude_lower=True,
                 exclude_upper=True):
    """
    Function checks if value is within given limits. If not then ValueError is raised.

    Parameters
    ----------
    value : float

    lower_limit : float, default = 0

    upper_limit : float, default = 1

    exclude_lower : bool, default = True
        If set to ``True`` then value must be larger than the lower limit.
        If ``False`` then value could be equal to the lower limit (and
        larger than it).

    exclude_upper : bool, default = True
        If set to ``True`` then value must be lower than the upper limit.
        If ``False`` then value must be equal or lower than the upper limit.

    Raises
    ------
    ValueError :
        Value is outside the limits range.
    """

    msg = (f'Value {value} is outside the range {lower_limit}:{upper_limit}. '
           f'Lower limit is excluded: {exclude_lower}, Upper limit is '
           f'excluded: {exclude_upper}.')

    if value == lower_limit:
        if lower_limit == upper_limit:
            raise ValueError('Provided value, lower and '
                             'upper limits are the same')

    # <=
    if exclude_lower:
        if value <= lower_limit:
            raise ValueError(msg)

    # <
    if value < lower_limit:
        raise ValueError(msg)

    # >=
    if exclude_upper:
        if value >= upper_limit:
            raise ValueError(msg)

    # >
    if value > upper_limit:
        raise ValueError(msg)
