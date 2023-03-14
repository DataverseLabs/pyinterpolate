"""
Area-to-area Poisson Kriging function.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

"""
import logging
import warnings
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.kriging.models.block.weight import weights_array, WeightedBlock2BlockSemivariance
from pyinterpolate.kriging.utils.kwarnings import ExperimentalFeatureWarning
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_poisson_kriging_data, prepare_pk_known_areas, \
    get_aggregated_point_support_values, get_distances_within_unknown
from pyinterpolate.processing.transform.transform import get_areal_values_from_agg, transform_ps_to_dict, sem_to_cov
from pyinterpolate.variogram import TheoreticalVariogram


def area_to_area_pk(semivariogram_model: TheoreticalVariogram,
                    blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                    point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                    unknown_block: np.ndarray,
                    unknown_block_point_support: np.ndarray,
                    number_of_neighbors: int,
                    raise_when_negative_prediction=True,
                    raise_when_negative_error=True,
                    log_process=True):
    """
    Function predicts areal value in an unknown location based on the area-to-area Poisson Kriging

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

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    log_process : bool, default=True
        Log process info and debug info.

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

    # Prepare Kriging Data
    # {known block id: [(unknown x, unknown y), [unknown val, known val, distance between points]]}
    # Transform point support to dict
    if isinstance(point_support, Dict):
        dps = point_support
    else:
        if log_process:
            logging.info('Point support is transformed to dictionary')
        dps = transform_ps_to_dict(point_support)

    # Get c0
    sill = semivariogram_model.sill

    # Check ids
    kriging_data = select_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_point_support_dict=dps,
        nn=number_of_neighbors,
        max_range=semivariogram_model.rang,
        direction=semivariogram_model.direction,
        angular_tolerance=5
    )

    b2b_semivariance = WeightedBlock2BlockSemivariance(semivariance_model=semivariogram_model)
    avg_semivariances = b2b_semivariance.calculate_average_semivariance(kriging_data)

    k = np.array(list(avg_semivariances.values()))
    k = sem_to_cov(k, sill)
    k = k.T
    k_ones = np.ones(len(k) + 1)
    k_ones[:-1] = k

    # Prepare blocks for calculation
    prepared_ids = list(avg_semivariances.keys())

    # {(known block id a, known block id b): [pt a val, pt b val, distance between points]}
    distances_between_known_areas = prepare_pk_known_areas(dps, prepared_ids)

    semivariances_between_known_areas = b2b_semivariance.calculate_average_semivariance(
        distances_between_known_areas)

    # Create array
    predicted = []
    for idx_a in prepared_ids:
        row = []
        for idx_b in prepared_ids:
            row.append(semivariances_between_known_areas[(idx_a, idx_b)])
        predicted.append(row)

    predicted = np.array(predicted)
    predicted = sem_to_cov(predicted, sill)

    # Add diagonal weights
    values = get_areal_values_from_agg(blocks, prepared_ids)
    aggregated_ps = get_aggregated_point_support_values(dps, prepared_ids)
    weights = weights_array(predicted.shape, values, aggregated_ps)
    weighted_and_predicted = predicted + weights

    # Prepare weights matrix
    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[weighted_and_predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    try:
        w = np.linalg.solve(weights, k_ones)
    except TypeError:
        if log_process:
            logging.debug('Wrong dtypes used for np.linalg.solve, casting to float.')
        weights = weights.astype(np.float)
        k_ones = k_ones.astype(np.float)
        w = np.linalg.solve(weights, k_ones)

    zhat = values.dot(w[:-1])

    # Calculate prediction error

    if isinstance(unknown_block[0], np.ndarray):
        u_idx = unknown_block[0][0]
    else:
        u_idx = unknown_block[0]

    if zhat < 0:
        if log_process:
            logging.debug(f'Prediction below 0 for area {u_idx}')
        if raise_when_negative_prediction:
            raise ValueError(f'Predicted value is {zhat} and it should not be lower than 0. Check your sampling '
                             f'grid, samples, number of neighbors or semivariogram model type.')

    sigmasq = np.matmul(w.T, k_ones)


    distances_within_unknown_block = get_distances_within_unknown(unknown_block_point_support)
    semivariance_within_unknown = b2b_semivariance.calculate_average_semivariance({
        u_idx: distances_within_unknown_block
    })[u_idx]

    covariance_within_unknown = sem_to_cov([semivariance_within_unknown], sill)[0]

    sigmasq = covariance_within_unknown - sigmasq

    if sigmasq < 0:
        if log_process:
            logging.debug(f'Variance Error below 0 for area {u_idx}')
        if raise_when_negative_error:
            raise ValueError(f'Predicted error value is {sigmasq} and it should not be lower than 0. '
                             f'Check your sampling grid, samples, number of neighbors or semivariogram model type.')
        sigma = 0
    else:
        sigma = np.sqrt(sigmasq)

    results = [u_idx, zhat, sigma]
    return results
