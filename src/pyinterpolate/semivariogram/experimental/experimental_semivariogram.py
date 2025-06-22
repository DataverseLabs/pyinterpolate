from typing import Union, List, Any, Dict

import numpy as np

from pyinterpolate.core.data_models.points import VariogramPoints
from pyinterpolate.core.validators.experimental_semivariance import \
    validate_semivariance_weights, validate_direction_and_tolerance, \
    validate_bins
from pyinterpolate.semivariogram.experimental.functions.directional import \
    directional_weighted_semivariance, from_ellipse, from_triangle, \
    from_ellipse_cloud, from_triangle_cloud
from pyinterpolate.semivariogram.experimental.functions.general import \
    omnidirectional_variogram, omnidirectional_semivariogram_cloud
from pyinterpolate.semivariogram.experimental.functions.semivariance import \
    semivariance_fn
from pyinterpolate.semivariogram.lags.lags import get_lags


def calculate_semivariance(ds: Union[np.ndarray, VariogramPoints],
                           step_size: float = None,
                           max_range: float = None,
                           direction: float = None,
                           tolerance: float = None,
                           dir_neighbors_selection_method: str = 't',
                           custom_bins: Union[np.ndarray, Any] = None,
                           custom_weights: np.ndarray = None) -> np.ndarray:
    """
    Calculates experimental semivariance.

    Parameters
    ----------
    ds : numpy array
        ``[x, y, value]``

    step_size : float
        The fixed distance between lags grouping point neighbors.

    max_range : float
        The maximum distance at which the semivariance is calculated.

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

        * "triangle" or "t", default method where point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", more accurate method but also slower.

    custom_bins : numpy array, optional
        Custom bins for semivariance calculation. If provided, then parameter
        ``step_size`` is ignored and ``max_range`` is set to the final bin
        distance.

    custom_weights : numpy array, optional
        Custom weights assigned to points.

    Returns
    -------
    semivariance : numpy array
        ``[lag, semivariance, number of point pairs]``

    Notes
    -----
    # Semivariance

    It is a measure of dissimilarity between points over distance.
    We assume that the close observations tend to be similar (see Tobler's
    Law). Distant observations are less and less similar up to the distance
    where the influence of one point value on the other is negligible.

    We calculate the empirical semivariance as:

    .. math:: g(h) = 0.5 * n(h)^(-1) * (SUM|i=1, n(h)|: [z(x_i + h) - z(x_i)]^2)

    where:

    - :math:`h`: lag,
    - :math:`g(h)`: empirical semivariance for lag :math:`h`,
    - :math:`n(h)`: number of point pairs within a specific lag,
    - :math:`z(x_i)`: point a (value of observation at point a),
    - :math:`z(x_i + h)`: point b in distance h from point a (value of
      observation at point b).

    As an output we get array of lags :math:`h`, semivariances :math:`g`
    and number of points within each lag :math:`n`.

    # Weighted Semivariance

    Sometimes, we need to weight each point by a specific factor.
    It is especially important for the semivariogram deconvolution and
    Poisson Kriging. The weighting factor could be the time effort for
    observation at a location (ecology) or population size at a specific
    block (public health). Implementation of the algorithm follows
    publications:

        1. A. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C:
        Comparison of model based geostatistical methods in ecology:
        application to fin whale spatial distribution in northwestern
        Mediterranean Sea. In Geostatistics Banff 2004 Volume 2.
        Edited by: Leuangthong O, Deutsch CV. Dordrecht, The Netherlands,
        Kluwer Academic Publishers; 2005:777-786.

        2. B. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C:
        Geostatistical modelling of spatial distribution of Balenoptera
        physalus in the northwestern Mediterranean Sea from sparse count data
        and heterogeneous observation efforts. Ecological Modelling 2006.

    We calculate the weighted empirical semivariance as:

    .. math:: g_w(h) = 0.5 * (SUM|i=1, n(h)|: w(h))^(-1) * (SUM|i=1, n(h)|: w(h) * z_w(h))

    .. math:: w(h) = [n(x_i) * n(x_i + h)] / [n(u_i) + n(u_i + h)]

    .. math:: z_w(h) = (z(x_i) - z(x_i + h))^2 - m'

    where:

    - :math:`h`: lag,
    - :math:`g_w(h)`: weighted empirical semivariance for lag :math:`h`,
    - :math:`n(h)`: number of point pairs within a specific lag,
    - :math:`z(x_i)`: point a (rate of specific process at point a),
    - :math:`z(x_i + h)`: point b in distance :math:`h` from point a
      (rate of specific process at point b),
    - :math:`n(x_i)`: denominator value size at point a (time, population ...),
    - :math:`n(x_i + h)`: denominator value size at point b in distance
      :math:`h` from point a,
    - :math:`m'`: weighted mean of rates.

    The output of weighted algorithm is the same as for non-weighted data:
    array of lags :math:`h`, semivariances :math:`g` and number of
    points within each lag :math:`n`.

    # Directional Semivariogram

    The assumption that our observations change in the same way in every
    direction is rarely true. Let's consider temperature. It changes from
    the equator to the poles so in the N-S and S-N axes. The good idea is to
    test the correlation of our observations in a few different directions.
    The main difference between an omnidirectional semivariogram and
    a directional semivariogram is that we take into account a different
    subset of neighbors:

    - Omnidirectional semivariogram: we test neighbors in a circle,
    - Directional semivariogram: we test neighbors within an ellipse,
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
    >>> semivariances = calculate_semivariance(
    ...    REFERENCE_INPUT,
    ...    step_size=STEP_SIZE,
    ...    max_range=MAX_RANGE)
    >>> print(semivariances[0])
    [ 1.     4.625 24.   ]
    """

    # Validation
    # Validate points
    if not isinstance(ds, VariogramPoints):
        ds = VariogramPoints(points=ds)

    # Validate bins
    validate_bins(step_size, max_range, custom_bins)

    # Validate custom_weights
    validate_semivariance_weights(ds.points, custom_weights)

    # Validate direction and tolerance
    validate_direction_and_tolerance(direction, tolerance)

    # Calculations
    # Get lags
    lags = get_lags(step_size, max_range, custom_bins)

    # Get semivariances
    if direction is not None and tolerance is not None:
        experimental_semivariances = directional_semivariance(
            ds.points,
            lags,
            direction,
            tolerance,
            dir_neighbors_selection_method,
            custom_weights,
            # as_point_cloud=False
        )
    else:
        experimental_semivariances = omnidirectional_semivariance(
            ds.points, lags, custom_weights, as_point_cloud=False
        )

    return experimental_semivariances


def directional_semivariance_cloud(points: np.ndarray,
                                   lags: Union[List, np.ndarray],
                                   direction: float,
                                   tolerance: float,
                                   method: str) -> Dict:
    """
    Function calculates directional semivariances.

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

    method : str
        Neighbors selection in a given direction. Available methods:

        * "triangle" or "t", default method where point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", more accurate method but also slower.

    Returns
    -------
    : Dict
        ``{lag: [semivariances]}``
    """
    output_semivariances = dict()

    if method == "e" or method == "ellipse":
        output_semivariances = from_ellipse_cloud(
            points,
            lags,
            direction,
            tolerance)
    elif method == "t" or method == "triangle":
        output_semivariances = from_triangle_cloud(
            points=points,
            lags=lags,
            direction=direction,
            tolerance=tolerance
        )

    return output_semivariances


def directional_semivariance(points: np.ndarray,
                             lags: Union[List, np.ndarray],
                             direction: float,
                             tolerance: float,
                             dir_neighbor_selection_method: str,
                             custom_weights: np.ndarray = None):
    """
    Function calculates directional semivariances.

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

    dir_neighbor_selection_method : str
        Neighbors selection in a given direction. Available methods:

        * "triangle" or "t", default method where point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", more accurate method but also slower.

    custom_weights : optional, Iterable
        Custom weights assigned to points.

    Returns
    -------
    : (numpy array)
      ``[lag, semivariance, number of point pairs]``
    """
    output_semivariances = np.array([])

    if custom_weights is None:
        if (dir_neighbor_selection_method == "e" or
            dir_neighbor_selection_method == "ellipse"):
            output_semivariances = from_ellipse(
                semivariance_fn,
                points,
                lags,
                direction,
                tolerance)
        elif (dir_neighbor_selection_method == "t" or
              dir_neighbor_selection_method == "triangle"):
            output_semivariances = from_triangle(
                fn=semivariance_fn,
                points=points,
                lags=lags,
                direction=direction,
                tolerance=tolerance
            )
    else:
        output_semivariances = directional_weighted_semivariance(
            points, lags, custom_weights, direction, tolerance
        )

    return output_semivariances


def omnidirectional_semivariance(points: np.ndarray,
                                 lags: Union[List, np.ndarray],
                                 custom_weights: np.ndarray,
                                 as_point_cloud: bool = False):
    """
    Function calculates the omnidirectional semivariances.

    Parameters
    ----------
    points : numpy array

    lags : Iterable

    custom_weights : Iterable

    as_point_cloud : bool
        Return semivariances as a point cloud.

    Returns
    -------
    semivariances : Union[numpy array, dict]
    """
    if not as_point_cloud:
        sorted_semivariances = omnidirectional_variogram(
            fn=semivariance_fn,
            points=points,
            lags=lags,
            custom_weights=custom_weights
        )
    else:

        if custom_weights is not None:
            # todo warning message - custom_weights not included
            pass

        sorted_semivariances = omnidirectional_semivariogram_cloud(
            points=points,
            lags=lags
        )

    return sorted_semivariances


def point_cloud_semivariance(ds: Union[np.ndarray, VariogramPoints],
                             step_size: float = None,
                             max_range: float = None,
                             direction: float = None,
                             tolerance: float = None,
                             dir_neighbors_selection_method: str = 't',
                             custom_bins: Union[np.ndarray, Any] = None,
                             custom_weights: np.ndarray = None) -> Dict:
    """
    Calculates experimental semivariance.

    Parameters
    ----------
    ds : numpy array
        ``[x, y, value]``

    step_size : float
        The fixed distance between lags grouping point neighbors.

    max_range : float
        The maximum distance at which the semivariance is calculated.

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

        * "triangle" or "t", default method where point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", more accurate method but also slower.

    custom_bins : numpy array, optional
        Custom bins for semivariance calculation. If provided, then parameter
        ``step_size`` is ignored and ``max_range`` is set to the final bin
        distance.

    custom_weights : numpy array, optional
        Custom weights assigned to points.

    Returns
    -------
    semivariance : Dict
        ``{lag: [semivariances], }``

    Notes
    -----
    # Semivariance

    It is a measure of dissimilarity between points over distance.
    We assume that the close observations tend to be similar (see Tobler's
    Law). Distant observations are less and less similar up to the distance
    where the influence of one point value on the other is negligible.

    We calculate the empirical semivariance as:

        (1)    g(h) = 0.5 * n(h)^(-1) * (
                      SUM|i=1, n(h)|: [z(x_i + h) - z(x_i)]^2
                      )

        where:
            h: lag,
            g(h): empirical semivariance for lag h,
            n(h): number of point pairs within a specific lag,
            z(x_i): point a (value of observation at point a),
            z(x_i + h): point b in distance h from point a (value of
              observation at point b).

    As an output we get array of lags h, semivariances g and number of
    points within each lag n.

    # Weighted Semivariance

    Sometimes, we need to weight each point by a specific factor.
    It is especially important for the semivariogram deconvolution and
    Poisson Kriging. The weighting factor could be the time effort for
    observation at a location (ecology) or population size at a specific
    block (public health). Implementation of the algorithm follows
    publications:

        1. A. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C:
        Comparison of model based geostatistical methods in ecology:
        application to fin whale spatial distribution in northwestern
        Mediterranean Sea. In Geostatistics Banff 2004 Volume 2.
        Edited by: Leuangthong O, Deutsch CV. Dordrecht, The Netherlands,
        Kluwer Academic Publishers; 2005:777-786.
        2. B. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C:
        Geostatistical modelling of spatial distribution of Balenoptera
        physalus in the northwestern Mediterranean Sea from sparse count data
        and heterogeneous observation efforts. Ecological Modelling 2006.

    We calculate the weighted empirical semivariance as:

        (2)    g_w(h) = 0.5 * (SUM|i=1, n(h)|: w(h))^(-1) * ...
                            * (SUM|i=1, n(h)|: w(h) * z_w(h))

        (3)    w(h) = [n(x_i) * n(x_i + h)] / [n(u_i) + n(u_i + h)]

        (4)    z_w(h) = (z(x_i) - z(x_i + h))^2 - m'

        where:
            h: lag,
            g_w(h): weighted empirical semivariance for lag h,
            n(h): number of point pairs within a specific lag,
            z(x_i): point a (rate of specific process at point a),
            z(x_i + h): point b in distance h from point a (rate of specific
              process at point b),
            n(x_i): denominator value size at point a (time, population ...),
            n(x_i + h): denominator value size at point b in distance h from
              point a,
            m': weighted mean of rates.

    The output of weighted algorithm is the same as for non-weighted data:
    array of lags h, semivariances g and number of points within each lag n.

    # Directional Semivariogram

    The assumption that our observations change in the same way in every
    direction is rarely true. Let's consider temperature. It changes from
    the equator to the poles so in the N-S and S-N axes. The good idea is to
    test the correlation of our observations in a few different directions.
    The main difference between an omnidirectional semivariogram and
    a directional semivariogram is that we take into account a different
    subset of neighbors:

       - Omnidirectional semivariogram: we test neighbors in a circle,
       - Directional semivariogram: we test neighbors within an ellipse,
         and one direction is major.

    Examples
    --------
    """

    # Validation
    # Validate points
    if not isinstance(ds, VariogramPoints):
        ds = VariogramPoints(points=ds)

    # Validate bins
    validate_bins(step_size, max_range, custom_bins)

    # Validate custom_weights
    validate_semivariance_weights(ds.points, custom_weights)

    # Validate direction and tolerance
    validate_direction_and_tolerance(direction, tolerance)

    # Calculations
    # Get lags
    lags = get_lags(step_size, max_range, custom_bins)

    # Get semivariances
    if direction is not None and tolerance is not None:

        if custom_weights is not None:
            # todo warning message - custom_weights not included
            pass

        experimental_semivariances = directional_semivariance_cloud(
            ds.points,
            lags,
            direction,
            tolerance,
            dir_neighbors_selection_method
        )
    else:
        experimental_semivariances = omnidirectional_semivariance(
            ds.points, lags, custom_weights, as_point_cloud=True
        )

    return experimental_semivariances
