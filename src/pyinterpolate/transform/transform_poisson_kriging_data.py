from typing import Union, Hashable, List

import numpy as np
import pandas as pd
from shapely import Point

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.point_support_distances import PointSupportDistance
from pyinterpolate.distance.angular import calculate_angular_difference


def angles_to_unknown_block(block_index: Union[str, Hashable],
                            point_support: PointSupport):
    """
    Function gets angles to unknown block.

    Parameters
    ----------
    block_index : str | Hashable
        Unique block index.

    point_support : PointSupport
        Point support data.

    Returns
    -------
    : numpy array
        Angles between the unknown block and other blocks.
    """
    if point_support.blocks.angles is None:
        angles = point_support.blocks.calculate_angles_between_rep_points(
            update=True
        )
    else:
        angles = point_support.blocks.angles

    block_angles = angles[block_index]

    return block_angles


def block_to_blocks_angles(block_id: Union[str, Hashable],
                           point_support: PointSupport,
                           direction: float):
    """
    Function calculates angles between given areas and the unknown area.

    Parameters
    ----------
    block_id : Union[str, Hashable]
        Unique block index.

    point_support : PointSupport
        Point support data.

    direction : float
        Expected direction.

    Returns
    -------
    : numpy array
        Difference between the angle from unknown block centroid to origin,
        and from other blocks representative points to origin.
    """

    angles = angles_to_unknown_block(
        block_index=block_id,
        point_support=point_support
    )
    angle_differences = calculate_angular_difference(
        angles=angles,
        expected_direction=direction
    )
    return angle_differences


def block_base_distances(
        block_id: Union[str, Hashable],
        point_support: PointSupport,
        point_support_distances: PointSupportDistance = None
) -> np.ndarray:
    """
    Function gets distances to the unknown block from other blocks.

    Parameters
    ----------
    block_id : Union[Hashable, str]

    point_support : PointSupport

    point_support_distances : PointSupportDistance
        Object with estimated weighted distances. If not given then it is
        assumed that function returns not weighted distances.

    Returns
    -------
    : numpy array
        Distances from the unknown block to other blocks.
    """
    if point_support_distances.weighted_block_to_block_distances is not None:
        return point_support_distances.weighted_block_to_block_distances.get(
            block_id
        )
    else:
        # Calc from coordinates
        if point_support.blocks_distances is None:
            dists = point_support.blocks.calculate_distances_between_rep_points(
                update=True
            )
        else:
            dists = point_support.blocks_distances

        distances = dists[block_id]
        return distances.to_numpy()


def set_blocks_dataset(block_id: Union[str, Hashable],
                       points: np.ndarray,
                       values: np.ndarray,
                       distances: np.ndarray,
                       blocks_indexes: np.ndarray = None,
                       angular_differences: np.ndarray = None,
                       angular_differences_column: str = "angular_differences",
                       blocks_indexes_column: str = "blocks_indexes",
                       core_block_id_column: str = "block_id",
                       distances_column: str = "distances",
                       points_column: str = "points",
                       values_column: str = "values") -> pd.DataFrame:
    """
    Function sets DataFrame for Poisson Kriging using blocks
    (representative points) data.

    Parameters
    ----------
    block_id : Union[str, Hashable]
        Unknown block ID.

    points : numpy array
        Representative coordinates of the known blocks.

    values : numpy array
        Aggregated values of the known blocks.

    distances : numpy array
        Distances between ``block_id`` and other blocks.

    blocks_indexes : optional, numpy array
        The list of the known blocks names.

    angular_differences : optional, numpy array
        Angular differences between origin and ``block_id`` representative
        point, and origin with other representative points.

    angular_differences_column : str, default = "angular_differences"

    blocks_indexes_column : str, default = "blocks_indexes"

    core_block_id_column : str, default = "block_id"

    distances_column : str, default = "distances"

    points_column : str, default = "points"

    values_column : str, default = "values"

    Returns
    -------
    : DataFrame
        Columns representing: other blocks, points, values, distances,
        angles, core block index.
    """

    if blocks_indexes is None:
        if not isinstance(points[0], Point):
            blocks_indexes = [Point([x[0], x[1]]) for x in points]

    ds = pd.DataFrame()
    ds[blocks_indexes_column] = blocks_indexes
    ds[points_column] = [tuple(x) for x in points]
    ds[values_column] = values

    if isinstance(distances, pd.Series):
        distances.name = distances_column
        ds[distances_column] = distances.to_numpy()
    else:
        ds[distances_column] = distances

    ds[angular_differences_column] = angular_differences

    ds[core_block_id_column] = block_id

    ds.set_index(blocks_indexes_column, inplace=True)
    return ds


def parse_kriging_input(unknown_points_and_values: np.ndarray,
                        known_blocks_id: List[Union[Hashable, str]],
                        known_points_and_values: List[np.ndarray],
                        distances: np.ndarray,
                        angle_diffs: np.ndarray = None,) -> pd.DataFrame:
    """
    Parses input into a DataFrame.

    Parameters
    ----------
    unknown_points_and_values : numpy array
        ``[x, y, z]``

    known_blocks_id : List
        Indexes of unknown blocks.

    known_points_and_values : List[numpy array]
        ``[[ux_1, uy_1, uz_1], ..., [ux_n, uy_n, uz_n]]``

    distances : numpy array
        Distances between ALL points from the unknown block and ALL points
        from the known blocks. Rows -> unknown block coordinates,
        columns -> known block coordinates.

    angle_diffs : numpy array, optional
        Angles between ALL points from the unknown block and ALL points from
        the known blocks. Rows -> unknown block coordinates,
        columns -> known block coordinates.

    Returns
    -------
    : pandas DataFrame
        Columns:
          * ``known_block_id``: index of the known block
          * ``kx``: x-coordinates of the known block's point support
          * ``ky``: y-coordinates of the known block's point support
          * ``k-value``: values of the known block's point support
          * ``ux``: x-coordinate of the unknown block's point support
          * ``uy``: y-coordinate of the unknown block's point support
          * ``u-value``: value of the unknown block's point support
          * ``distance``: distance between points ``(kx, ky)`` and ``(ux, uy)``
          * ``angular_difference``: optional, angle between points
            ``(kx, ky)`` and ``(ux, uy)``.
    """

    data = []
    columns = [
        'known_block_id',
        'kx',
        'ky',
        'k_value',
        'ux',
        'uy',
        'u_value',
        'distance',
        'angular_difference'
    ]

    for pos, k_block_idx in enumerate(known_blocks_id):
        _dists_arr = distances[pos]
        _coords = known_points_and_values[pos]
        for row, _dists in enumerate(_dists_arr):
            for idx_coord, coord in enumerate(_coords):

                ds = [
                        k_block_idx,
                        coord[0],
                        coord[1],
                        coord[2],
                        unknown_points_and_values[row, 0],
                        unknown_points_and_values[row, 1],
                        unknown_points_and_values[row, 2],
                        _dists[idx_coord]
                    ]

                if angle_diffs is not None:
                    ds.append(
                        angle_diffs[pos][row][idx_coord]
                    )
                else:
                    ds.append(None)

                data.append(ds)

    df = pd.DataFrame(data=data, columns=columns)
    return df
