from typing import Union, Dict

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.kriging.models.block.weight import WeightedBlock2BlockSemivariance,\
    WeightedBlock2PointSemivariance, add_ones, weights_array
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_poisson_kriging_data, prepare_pk_known_areas, \
    get_aggregated_point_support_values
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, get_areal_values_from_agg
from pyinterpolate.variogram import TheoreticalVariogram


def area_to_point_pk(semivariogram_model: TheoreticalVariogram,
                     blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                     point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                     unknown_block: np.ndarray,
                     unknown_block_point_support: np.ndarray,
                     number_of_neighbors: int):
    """
    Function predicts areal value in a unknown location based on the area-to-area Poisson Kriging

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
                          Regularized variogram.

    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
             Blocks with aggregated data.
             * Blocks: Blocks() class object.
             * GeoDataFrame and DataFrame must have columns: centroid.x, centroid.y, ds, index.
               Geometry column with polygons is not used and optional.
             * numpy array: [[block index, centroid x, centroid y, value]].

    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
                    * Dict: {block id: [[point x, point y, value]]}
                    * numpy array: [[block id, x, y, value]]
                    * DataFrame and GeoDataFrame: columns={x, y, ds, index}
                    * PointSupport

    unknown_block : numpy array
                    [index, centroid.x, centroid.y]

    unknown_block_point_support : numpy array
                                  Points within block [[x, y, point support value]]

    number_of_neighbors : int
                          The minimum number of neighbours that potentially affect block.


    Returns
    -------
    results : List
              [(unknown point coordinates), prediction, error]

    """
    # Get total point-support value of the unknown area
    tot_unknown_value = np.sum(unknown_block_point_support[:, -1])

    # Transform point support to dict
    if isinstance(point_support, Dict):
        dps = point_support
    else:
        dps = transform_ps_to_dict(point_support)

    # Prepare Kriging Data
    # {known block id: [unknown_pt_idx_coordinates, known pt val, unknown pt val, distance between points]}
    kriging_data = select_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_point_support_dict=dps,
        nn=number_of_neighbors)

    prepared_ids = list(kriging_data.keys())

    # Get block to block weighted semivariances
    b2b_semivariance = WeightedBlock2BlockSemivariance(semivariance_model=semivariogram_model)

    # Get block to point weighted semivariances
    b2p_semivariance = WeightedBlock2PointSemivariance(semivariance_model=semivariogram_model)
    # array(
    #     [unknown point 1 (n) semivariance against point support from block 1,
    #      unknown point 2 (n+1) semivariance against point support from block 1,
    #      ...],
    #     [unknown point 1 (n) semivariance against point support from block 2,
    #      unknown point 2 (n+1) semivariance against point support from block 2,
    #      ...],
    # )
    avg_b2p_semivariances = add_ones(b2p_semivariance.calculate_average_semivariance(kriging_data))

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
    transposed = avg_b2p_semivariances.T

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

        if (zhat < 0) or (zhat == np.nan):
            zhat = 0

        point_pop = unknown_block_point_support[idx, -1]
        zhat = (zhat * point_pop) / tot_unknown_value

        # Calculate error
        sigmasq = (w.T * point)[0]
        if sigmasq < 0:
            # TODO: Alert user, change number of neighbors, do spatial resampling etc.
            sigma = 0
        else:
            sigma = np.sqrt(sigmasq)

        predicted_points.append([(analyzed_pts[0], analyzed_pts[1]), zhat, sigma])

    return np.array(predicted_points)
