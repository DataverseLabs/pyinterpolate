"""
Area-to-point Poisson Kriging function.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
from typing import Union, Dict
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.kriging.models.block.weight import WeightedBlock2BlockSemivariance,\
    WeightedBlock2PointSemivariance, add_ones, weights_array
from pyinterpolate.kriging.utils.kwarnings import ExperimentalFeatureWarning
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_poisson_kriging_data, prepare_pk_known_areas, \
    get_aggregated_point_support_values
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, get_areal_values_from_agg, sem_to_cov
from pyinterpolate.variogram import TheoreticalVariogram


def area_to_point_pk(semivariogram_model: TheoreticalVariogram,
                     blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                     point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                     unknown_block: np.ndarray,
                     unknown_block_point_support: np.ndarray,
                     number_of_neighbors: int,
                     max_range=None,
                     raise_when_negative_prediction=True,
                     raise_when_negative_error=True,
                     err_to_nan=True):
    """
    Function predicts areal value in the unknown location based on the area-to-area Poisson Kriging

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

    max_range : float , default=None
        The maximum distance to search for a neighbors, if ``None`` given then algorithm uses
        the theoretical variogram's range.

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    err_to_nan : bool, default=True
        ``ValueError`` to ``NaN``.


    Returns
    -------
    results : List
        ``[(unknown point coordinates), prediction, error]``

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

    logging.info("POISSON KRIGING: AREA-TO-POINT | Operation starts")
    # Get total point-support value of the unknown area
    tot_unknown_value = np.sum(unknown_block_point_support[:, -1])

    # Transform point support to dict
    if isinstance(point_support, Dict):
        dps = point_support
    else:
        dps = transform_ps_to_dict(point_support)

    # Prepare Kriging Data
    # {known block id: [unknown_pt_idx_coordinates, known pt val, unknown pt val, distance between points]}

    if max_range is None:
        rng = semivariogram_model.rang
    else:
        rng = max_range

    # Get sill to calc cov
    sill = semivariogram_model.sill

    kriging_data = select_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_point_support_dict=dps,
        nn=number_of_neighbors,
        max_range=rng
    )

    prepared_ids = list(kriging_data.keys())
    logging.info("POISSON KRIGING: AREA-TO-POINT | Kriging data has been prepared")

    # Get block to block weighted semivariances
    b2b_semivariance = WeightedBlock2BlockSemivariance(semivariance_model=semivariogram_model)
    logging.info("POISSON KRIGING: AREA-TO-POINT | Block to block semivariances are calculated")

    # Get block to point weighted semivariances
    b2p_semivariance = WeightedBlock2PointSemivariance(semivariance_model=semivariogram_model)
    logging.info("POISSON KRIGING: AREA-TO-POINT | Block to point semivariances are calculated")
    # array(
    #     [unknown point 1 (n) semivariance against point support from block 1,
    #      unknown point 2 (n+1) semivariance against point support from block 1,
    #      ...],
    #     [unknown point 1 (n) semivariance against point support from block 2,
    #      unknown point 2 (n+1) semivariance against point support from block 2,
    #      ...],
    # )
    # Transform to covariances
    avg_b2p_covariances = add_ones(
        sem_to_cov(
            b2p_semivariance.calculate_average_semivariance(kriging_data),
            sill
        )
    )

    # {(known block id a, known block id b): [pt a val, pt b val, distance between points]}
    distances_between_known_areas = prepare_pk_known_areas(dps, prepared_ids)

    semivariances_between_known_areas = b2b_semivariance.calculate_average_semivariance(
        distances_between_known_areas)
    logging.info("POISSON KRIGING: AREA-TO-POINT | Average semivariance between areas is estimated")
    logging.info("POISSON KRIGING: AREA-TO-POINT | Kriging system weights preparation")
    # Create array
    predicted = []
    for idx_a in prepared_ids:
        row = []
        for idx_b in prepared_ids:
            row.append(semivariances_between_known_areas[(idx_a, idx_b)])
        predicted.append(row)

    predicted = np.array(predicted)

    # Transform to covariances
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

    # Solve Kriging systems
    # Each point of unknown area represents different kriging system

    # Transpose points matrix
    transposed = avg_b2p_covariances.T
    logging.info("POISSON KRIGING: AREA-TO-POINT | Kriging system weights has been prepared")

    predicted_points = []

    for idx, point in enumerate(transposed):

        analyzed_pts = unknown_block_point_support[idx, :-1]

        try:
            w = np.linalg.solve(weights, point)
        except TypeError:
            weights = weights.astype(np.float)
            point = point.astype(np.float)
            w = np.linalg.solve(weights, point)

        zhat = values.dot(w[:-1])

        if zhat < 0:
            if raise_when_negative_prediction:
                logging.info(f"POISSON KRIGING: AREA-TO-POINT | Negative prediction for point {point}")
                if err_to_nan:
                    logging.info(f"POISSON KRIGING: AREA-TO-POINT | Negative prediction - NaN zhat and error")
                    predicted_points.append([(analyzed_pts[0], analyzed_pts[1]), np.nan, np.nan])
                    continue
                else:
                    raise ValueError(f'Predicted value is {zhat} and it should not be lower than 0. Check your '
                                     f'sampling grid, samples, number of neighbors or semivariogram model type.')

        point_pop = unknown_block_point_support[idx, -1]
        zhat = (zhat * point_pop) / tot_unknown_value

        # Calculate error
        sigmasq = np.matmul(w.T, point)
        if sigmasq < 0:
            if raise_when_negative_error:
                logging.info(f"POISSON KRIGING: AREA-TO-POINT | Negative variance error for point {point}")
                if err_to_nan:
                    logging.info(f"POISSON KRIGING: AREA-TO-POINT | Negative error - NaN error")
                    predicted_points.append([(analyzed_pts[0], analyzed_pts[1]), zhat, np.nan])
                    continue
                else:
                    raise ValueError(f'Predicted error value is {sigmasq} and it should not be lower than 0. '
                                     f'Check your sampling grid, samples, number of neighbors or semivariogram model '
                                     f'type.')
            else:
                sigma = 0
        else:
            sigma = np.sqrt(sigmasq)

        predicted_points.append([(analyzed_pts[0], analyzed_pts[1]), zhat, sigma])

    logging.info(f"POISSON KRIGING: AREA-TO-POINT | Process ends")

    return predicted_points
