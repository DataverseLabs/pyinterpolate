"""
Centroid-based Poisson Kriging function.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.kriging.models.block.weight import weights_array
from pyinterpolate.kriging.utils.kwarnings import ExperimentalFeatureWarning
from pyinterpolate.kriging.utils.process import solve_weights
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_centroid_poisson_kriging_data
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, sem_to_cov
from pyinterpolate.variogram import TheoreticalVariogram


def centroid_poisson_kriging(semivariogram_model: TheoreticalVariogram,
                             blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                             point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                             unknown_block: np.ndarray,
                             unknown_block_point_support: np.ndarray,
                             number_of_neighbors: int,
                             is_weighted_by_point_support=True,
                             raise_when_negative_prediction=True,
                             raise_when_negative_error=True,
                             allow_approximate_solutions=False) -> List:
    """
    Function performs centroid-based Poisson Kriging of blocks (areal) data.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        A fitted variogram.

    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
        Blocks with aggregated data.
          * ``Blocks``: ``Blocks()`` class object.
          * ``GeoDataFrame`` and ``DataFrame`` must have columns: ``centroid_x, centroid_y, ds, index``.
            Geometry column with polygons is not used.
          * ``numpy array``: ``[[block index, centroid x, centroid y, value]]``.

    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
        The point support of polygons.
          * ``Dict``: ``{block id: [[point x, point y, value]]}``,
          * ``numpy array``: ``[[block id, x, y, value]]``,
          * ``DataFrame`` and ``GeoDataFrame``: ``columns={x_col, y_col, ds, index}``,
          * ``PointSupport``.

    unknown_block : numpy array
        ``[index, centroid x, centroid y]``

    unknown_block_point_support : numpy array
        Points within block ``[[x, y, point support value]]``

    number_of_neighbors : int
        The minimum number of neighbours that can potentially affect block.

    is_weighted_by_point_support : bool, default = True
        Are distances between blocks weighted by the point support?

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    results : List
        ``[unknown block index, prediction, error]``

    Raises
    ------
    ValueError
        Prediction or prediction error are negative.

    Warns
    -----
    ExperimentalFeatureWarning
        Directional Kriging is in early-phase and may contain bugs.

    """
    # Warnings area
    if semivariogram_model.direction is not None:
        exp_warning_msg = 'Directional Poisson Kriging is an experimental feature. Use it at your own responsibility!'
        warnings.warn(ExperimentalFeatureWarning(exp_warning_msg).__str__())

    # Get data: [block id, cx, cy, value, distance to unknown, aggregated point support sum]
    if isinstance(point_support, Dict):
        dps = point_support
    else:
        dps = transform_ps_to_dict(point_support)

    # Kriging data
    # [[cx, cy, value, distance to unknown, angles, aggregated point support sum]]
    kriging_data = select_centroid_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_blocks=blocks,
        k_point_support_dict=dps,
        nn=number_of_neighbors,
        max_range=semivariogram_model.rang,
        weighted=is_weighted_by_point_support,
        direction=semivariogram_model.direction
    )

    sill = semivariogram_model.sill

    distances_column_index = 3
    values_column_index = 2
    weights_column_index = 5

    n = len(kriging_data)
    distances = kriging_data[:, distances_column_index]
    values = kriging_data[:, values_column_index]

    partial_semivars = semivariogram_model.predict(distances)
    pcovars = sem_to_cov(partial_semivars, sill)
    covars = np.ones(len(pcovars) + 1)
    covars[:-1] = pcovars
    covars = covars.transpose()

    # Distances between known blocks
    coordinates = kriging_data[:, :values_column_index]
    block_distances = calc_point_to_point_distance(coordinates).flatten()
    known_blocks_semivars = semivariogram_model.predict(block_distances)
    predicted = np.array(known_blocks_semivars.reshape(n, n))
    predicted = sem_to_cov(predicted, sill)

    # Add diagonal weights to predicted semivars array
    weights = weights_array(predicted.shape, values, kriging_data[:, weights_column_index])
    weighted_and_predicted = predicted + weights

    # Prepare matrix for solving kriging system
    ones_col = np.ones((weighted_and_predicted.shape[0], 1))
    weighted_and_predicted = np.c_[weighted_and_predicted, ones_col]
    ones_row = np.ones((1, weighted_and_predicted.shape[1]))
    ones_row[0][-1] = 0
    kriging_weights = np.r_[weighted_and_predicted, ones_row]

    # Solve Kriging system
    try:
        output_weights = solve_weights(kriging_weights, covars, allow_approximate_solutions)
    except np.linalg.LinAlgError as _:
        msg = 'Singular matrix in Kriging system detected, check if you have duplicated coordinates ' \
              'in the ``known_locations`` variable.'
        raise RuntimeError(msg)

    zhat = values.dot(output_weights[:-1])

    if raise_when_negative_prediction:
        if zhat < 0:
            raise ValueError(f'Predicted value is {zhat} and it should not be lower than 0. Check your sampling '
                             f'grid, samples, number of neighbors or semivariogram model type.')

    sigmasq = np.matmul(output_weights.T, covars)

    if sigmasq < 0:
        if raise_when_negative_error:
            raise ValueError(f'Predicted error value is {sigmasq} and it should not be lower than 0. '
                             f'Check your sampling grid, samples, number of neighbors or semivariogram model type.')
        sigma = np.nan
    else:
        sigma = np.sqrt(sigmasq)

    # Prepare output
    if isinstance(unknown_block[0], np.ndarray):
        u_idx = unknown_block[0][0]
    else:
        u_idx = unknown_block[0]

    results = [u_idx, zhat, sigma]
    return results
