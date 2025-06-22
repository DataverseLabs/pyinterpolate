from typing import Union, List, Any

import numpy as np

from pyinterpolate.core.data_models.points import VariogramPoints
from pyinterpolate.core.validators.experimental_semivariance import \
    validate_bins, validate_direction_and_tolerance
from pyinterpolate.semivariogram.experimental.functions.covariance import \
    covariance_fn
from pyinterpolate.semivariogram.experimental.functions.directional import \
    from_ellipse, from_triangle
from pyinterpolate.semivariogram.experimental.functions.general import \
    omnidirectional_variogram
from pyinterpolate.semivariogram.lags.lags import get_lags


def calculate_covariance(ds: Union[np.ndarray, VariogramPoints],
                         step_size: float = None,
                         max_range: float = None,
                         direction: float = None,
                         tolerance: float = None,
                         dir_neighbors_selection_method: str = 't',
                         custom_bins: Union[Any, np.ndarray] = None
                         ) -> np.ndarray:
    """
    Calculates experimental covariance.

    Parameters
    ----------
    ds : numpy array
        ``[x, y, value]``

    step_size : float
        The fixed distance between lags grouping point neighbors.

    max_range : float
        The maximum distance at which the covariance is calculated.

    direction : float, optional
        Direction of covariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional covariogram.

    dir_neighbors_selection_method : str, default = 't'
        Neighbors selection in a given direction. Available methods:

        * "triangle" or "t", default method where a point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", more accurate method but also slower.

    custom_bins : numpy array, optional
        Custom bins for covariance calculation. If provided, then parameter
        ``step_size`` is ignored and ``max_range`` is set to the final bin
        distance.

    Returns
    -------
    covariance : numpy array
        ``[lag, covariance, number of point pairs]``

    Notes
    -----
    # Covariance

    It is a measure of similarity between points over different distances.
    We assume that the close observations tend to be similar
    (recall the Tobler's Law).

    We calculate the empirical covariance as:

    .. math:: covariance = 1 / (N) * SUM(i=1, N) [z(x_i + h) * z(x_i)] - u^2

    where:

    - :math:`N` - number of observation pairs,
    - :math:`h` - distance (lag),
    - :math:`z(x_i)` - value at location :math:`z_i`,
    - :math:`(x_i + h)` - location at a distance :math:`h` from :math:`x_i`,
    - :math:`u` - average value of observations at a given lag distance.

    As an output we get array of lags :math:`h`, covariances :math:`c` and
    number of points within each lag :math:`n`.

    # Directional Covariogram

    The assumption that our observations change in the same way in every
    direction is rarely true. Let's consider temperature. It changes from
    the equator to the poles so in the N-S and S-N axes. The good idea is to
    test the correlation of our observations in a few different directions.
    The main difference between an omnidirectional covariogram and
    a directional covariogram is that we take into account a different
    subset of neighbors:

    - Omnidirectional covariogram: we test neighbors in a circle,
    - Directional covariogram: we test neighbors within an ellipse,
      and one direction is major.

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
    >>> covariances = calculate_covariance(
    ...     REFERENCE_INPUT, STEP_SIZE, MAX_RANGE
    ... )
    >>> print(covariances[0][0])
    [ 1.         -0.54340278 24.        ]
    >>> print(covariances[1])
    4.2485207100591715
    """

    # Validation
    # Validate points
    if not isinstance(ds, VariogramPoints):
        ds = VariogramPoints(points=ds)

    # Validate bins
    validate_bins(step_size, max_range, custom_bins)

    # Validate direction and tolerance
    validate_direction_and_tolerance(direction, tolerance)

    # Calculations
    # Get lags
    lags = get_lags(step_size, max_range, custom_bins)

    # Get covariances
    if direction is not None and tolerance is not None:
        experimental_covariances = directional_covariance(
            ds.points,
            lags,
            direction,
            tolerance,
            dir_neighbors_selection_method
        )
    else:
        experimental_covariances = omnidirectional_covariance(
            ds.points, lags
        )

    return experimental_covariances


def directional_covariance(points: np.ndarray,
                           lags: Union[List, np.ndarray],
                           direction: float,
                           tolerance: float,
                           dir_neighbors_selection_method: str):
    """
    Function calculates directional covariances.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    lags : numpy array

    direction : float, optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    dir_neighbors_selection_method : str, default = 't'
        Neighbors selection in a given direction. Available methods:

          * "triangle" or "t", default method where a point neighbors are
            selected from a triangular area,
          * "ellipse" or "e", more accurate method but also slower.

    Returns
    -------
    : (numpy array)
      ``[lag, covariance, number of point pairs]``
    """

    output_covariances = np.array([])

    if (dir_neighbors_selection_method == "e" or
        dir_neighbors_selection_method == "ellipse"):
        output_covariances = from_ellipse(covariance_fn,
                                          points,
                                          lags,
                                          direction,
                                          tolerance)
    elif (dir_neighbors_selection_method == "t" or
          dir_neighbors_selection_method == "triangle"):
        output_covariances = from_triangle(covariance_fn,
                                           points,
                                           lags,
                                           direction,
                                           tolerance)

    return output_covariances


def omnidirectional_covariance(points: np.array, lags: np.array) -> np.array:
    """Function calculates covariance from given points.

    Parameters
    ----------
    points : numpy array

    lags : Iterable

    Returns
    -------
    covariances : numpy array
    """

    sorted_covariances = omnidirectional_variogram(
        fn=covariance_fn,
        points=points,
        lags=lags,
        custom_weights=None
    )

    return sorted_covariances
