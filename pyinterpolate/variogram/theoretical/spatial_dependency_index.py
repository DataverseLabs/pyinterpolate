from typing import Tuple


def calculate_spatial_dependence_index(nugget: float, sill: float) -> Tuple:
    """
    Function estimates spatial dependence index and its ratio.

    Parameters
    ----------
    nugget : float

    sill : float

    Returns
    -------
    : Tuple[float, str]
        ratio, descriptive spatial dependency strength

    Raises
    ------
    ValueError
        Nugget is equal to zero.

    References
    ----------
    [1] CAMBARDELLA, C.A.; MOORMAN, T.B.; PARKIN, T.B.; KARLEN, D.L.; NOVAK, J.M.; TURCO, R.F.; KONOPKA, A.E.
    Field-scale variability of soil properties in central Iowa soils. Soil Science Society of America Journal,
    v. 58, n. 5, p. 1501-1511, 1994.
    """

    if nugget == 0:
        raise ValueError('Nugget cannot be set to 0 to calculate spatial dependence index')

    ratio = (nugget / sill) * 100

    if ratio < 25:
        spatial_dependency = 'strong'
    elif ratio < 75:
        spatial_dependency = 'moderate'
    elif ratio < 95:
        spatial_dependency = 'weak'
    else:
        spatial_dependency = 'no spatial dependency'

    return ratio, spatial_dependency
