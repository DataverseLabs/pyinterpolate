"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""

from typing import Dict, Union, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


from scipy.spatial.distance import cdist

from pyinterpolate.processing.preprocessing.blocks import PointSupport
from pyinterpolate.processing.transform.transform import point_support_to_dict, block_dataframe_to_dict


def _calc_b2b_dist_from_array(blocks: np.ndarray) -> Dict:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : numpy array
             [[block id, point x, point y, value]]

    Returns
    -------
    block_distances : Dict
                      {block id : [distances to other blocks]}. Block ids in the order from the list of
                      distances.
    """

    block_keys = np.unique(blocks[:, 0])
    block_distances = dict()
    for k_i in block_keys:
        i_block = blocks[blocks[:, 0] == k_i][:, 1:]
        distances = []
        for k_j in block_keys:
            j_block = blocks[blocks[:, 0] == k_j][:, 1:]
            if k_i == k_j:
                distances.append(0)
            else:
                value = _calculate_block_to_block_distance(i_block, j_block)
                distances.append(value)
        block_distances[k_i] = distances

    return block_distances


def _calc_b2b_dist_from_dataframe(blocks: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Dict:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : Union[pd.DataFrame, gpd.GeoDataFrame]
             DataFrame and GeoDataFrame: columns={x, y, ds, index}

    Returns
    -------
    block_distances : Dict
                      {block id : [distances to other blocks]}. Block ids in the order from the list of
                      distances.
    """

    expected_cols = {'x', 'y', 'ds', 'index'}

    if not expected_cols.issubset(set(blocks.columns)):
        raise KeyError(f'Given dataframe doesnt have all expected columns {expected_cols}. '
                       f'It has {blocks.columns} instead.')

    dsdict = block_dataframe_to_dict(blocks)

    bdists = _calc_b2b_dist_from_dict(dsdict)

    return bdists


def _calc_b2b_dist_from_dict(blocks: Dict) -> Dict:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : Dict
             Dict: {block id: [[point x, point y, value]]}

    Returns
    -------
    block_distances : Dict
                      {block id : [distances to other blocks]}. Block ids in the order from the list of
                      distances.
    """

    block_keys = list(blocks.keys())
    block_distances = dict()
    for k_i in block_keys:
        i_block = blocks[k_i]
        distances = []
        for k_j in block_keys:
            j_block = blocks[k_j]
            if k_i == k_j:
                distances.append(0)
            else:
                value = _calculate_block_to_block_distance(i_block, j_block)
                distances.append(value)
        block_distances[k_i] = distances

    return block_distances


def _calc_b2b_dist_from_ps(blocks: PointSupport) -> Dict:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : PointSupport

    Returns
    -------
    block_distances : Dict
                      {block id : [distances to other blocks]}. Block ids in the order from the list of
                      distances.
    """
    dsdict = point_support_to_dict(point_support=blocks)
    block_distances = _calc_b2b_dist_from_dict(dsdict)
    return block_distances


def calc_block_to_block_distance(blocks: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]) -> Dict:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
        The point support of polygons.
          * ``Dict``: ``{block id: [[point x, point y, value]]}``,
          * ``numpy array``: ``[[block id, x, y, value]]``,
          * ``DataFrame`` and ``GeoDataFrame``: ``columns={x, y, ds, index}``,
          * ``PointSupport``.


    Returns
    -------
    block_distances : Dict
        Ordered block ids (the order from the list of distances): {block id : [distances to other]}.

    Raises
    ------
    TypeError
        Wrong input's data type.
    """

    if isinstance(blocks, Dict):
        block_distances = _calc_b2b_dist_from_dict(blocks)
    elif isinstance(blocks, np.ndarray):
        block_distances = _calc_b2b_dist_from_array(blocks)
    elif isinstance(blocks, pd.DataFrame) or isinstance(blocks, gpd.GeoDataFrame):
        block_distances = _calc_b2b_dist_from_dataframe(blocks)
    elif isinstance(blocks, PointSupport):
        block_distances = _calc_b2b_dist_from_ps(blocks)
    else:
        raise TypeError(f'Blocks data type {type(blocks)} not recognized. You may use PointSupport,'
                        f' Geopandas GeoDataFrame, Pandas DataFrame or numpy array. See docs.')

    return block_distances


def _calculate_block_to_block_distance(block_1: np.ndarray, block_2: np.ndarray) -> float:
    """Function calculates distance between two blocks based on how they are divided (into the point support grid).

    Parameters
    ----------
    block_1 : numpy array

    block_2 : numpy array

    Returns
    -------
    weighted_distances : float
                         Weighted distance between blocks.

    Notes
    -----
    The weighted distance between blocks is derived from the equation:

    $$d(v_{a}, v_{b})=\frac{1}{\sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s}) n(u_{s'})} *
        \sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s})n(u_{s'})||u_{s}-u_{s'}||$$

    where:
    $P_{a}$ and $P_{b}$: number of points $u_{s}$ and $u_{s'}$ used to discretize the two units $v_{a}$ and $v_{b}$,
    $n(u_{s})$ and $n(u_{s'})$ - population size in the cells $u_{s}$ and $u_{s'}$.

    References
    ----------
    .. [1] Goovaerts, P. Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units.
           Math Geosci 40, 101–128 (2008). https://doi.org/10.1007/s11004-007-9129-1
    """

    a_shape = block_1.shape[0]
    b_shape = block_2.shape[0]
    ax = block_1[:, 0].reshape(1, a_shape)
    bx = block_2[:, 0].reshape(b_shape, 1)
    dx = ax - bx
    ay = block_1[:, 1].reshape(1, a_shape)
    by = block_2[:, 1].reshape(b_shape, 1)
    dy = ay - by
    aval = block_1[:, -1].reshape(1, a_shape)
    bval = block_2[:, -1].reshape(b_shape, 1)
    w = aval * bval

    dist = np.sqrt(dx ** 2 + dy ** 2, dtype=float, casting='unsafe')

    wdist = dist * w
    distances_sum = np.sum(wdist) / np.sum(w)
    return distances_sum


def _calc_angle_between_points(v1, v2, origin):
    ang1 = np.arctan2(v1[1] - origin[1], v1[0] - origin[0])
    ang2 = np.arctan2(v2[:, 1] - origin[1], v2[:, 0] - origin[0])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def _calc_angle_from_origin(vec, origin):
    ys = vec[:, 1] - origin[1]
    xs = vec[:, 0] - origin[0]
    ang = np.arctan2(ys, xs)
    return np.rad2deg(ang % (2 * np.pi))


def calc_angles_between_points(vec1, vec2, origin=None):
    """
    Function calculates distances between two groups of points as their cross product.

    Parameters
    ----------
    vec1 : numpy array
        The first set of coordinates.

    vec2 : numpy array
        The second set of coordinates.

    origin : Iterable, optional
        The coordinates (x, y) of origin.

    Returns
    -------
    angles : numpy array
        An array with angles between all points from ``vec1`` to all points from ``vec2``, where rows are angles between
         points from ``vec1`` to points from ``vec2`` (columns).
    """
    angles = []

    for point in vec1:
        row = calc_angles(vec2, origin=point)
        angles.append(row.flatten())

    angles = np.array(angles).flatten()
    return angles


def calc_angles(points_b: Iterable, point_a: Iterable = None, origin: Iterable = None):
    """
    Function calculates angles between points and origin or between vectors from origin to points and a vector from
    a specific point to origin.

    Parameters
    ----------
    points_b : numpy array
        Other point coordinates.

    point_a : Iterable
        The point coordinates, default is equal to (0, 0).

    origin : Iterable
        The origin coordinates, default is (0, 0).

    Returns
    -------
    angles : numpy array
        Angles from the ``points_b`` to origin, or angles between vectors ``points_b`` to origin and ``point_a``
        to origin.
    """

    if origin is None:
        origin = np.array((0, 0))
    else:
        if not isinstance(origin, np.ndarray):
            origin = np.array(origin)

    if not isinstance(points_b, np.ndarray):
        points_b = np.array(points_b)

    if len(points_b.shape) == 1:
        points_b = points_b[np.newaxis, ...]

    if point_a is None:
        angles = _calc_angle_from_origin(points_b, origin)
    else:
        angles = _calc_angle_between_points(point_a, points_b, origin)

    return angles


def calc_point_to_point_distance(points_a, points_b=None):
    """Function calculates distances between two group of points of a single group to itself.

    Parameters
    ----------
    points_a : numpy array
        The point coordinates.

    points_b : numpy array, default=None
        Other point coordinates. If provided then algorithm calculates distances between ``points_a`` against
        ``points_b``.

    Returns
    -------
    distances : numpy array
        The distances from each point from the ``points_a`` to other point (from the same ``points_a`` or from the
        other set of points ``points_b``).
    """

    if points_b is None:
        distances = cdist(points_a, points_a, 'euclidean')
    else:
        distances = cdist(points_a, points_b, 'euclidean')
    return distances


def calculate_angular_distance(angles: np.ndarray, expected_direction: float) -> np.ndarray:
    """
    Function calculates minimal direction between one vector and other vectors.

    Parameters
    ----------
    angles : numpy array
        The array with the direction to the origin of each point.

    expected_direction : float
        The variogram direction in degrees.

    Returns
    -------
    angular_distances : numpy array
        Minimal direction from ``expected_direction`` to other angles.
    """

    # We should select angles equal to the expected direction
    # and 180 degrees from it

    expected_direction_rad = np.deg2rad(expected_direction)
    r_angles = np.deg2rad(angles)
    norm_a = r_angles - expected_direction_rad
    deg_norm_a = np.abs(np.rad2deg(norm_a % (2 * np.pi)))
    norm_b = expected_direction_rad - r_angles
    deg_norm_b = np.abs(np.rad2deg(norm_b % (2 * np.pi)))
    normalized_angular_dists = np.minimum(deg_norm_a, deg_norm_b)

    return normalized_angular_dists
