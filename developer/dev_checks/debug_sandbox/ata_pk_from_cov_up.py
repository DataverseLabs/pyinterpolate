from datetime import datetime
from typing import Dict, Union

import numpy as np
import logging

import pandas as pd
import geopandas as gpd

from pyinterpolate.kriging.models.block.area_to_area_poisson_kriging import area_to_area_pk
from pyinterpolate.kriging.models.block.weight import WeightedBlock2BlockSemivariance, weights_array
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_poisson_kriging_data, prepare_pk_known_areas, \
    get_aggregated_point_support_values, get_distances_within_unknown
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, sem_to_cov, get_areal_values_from_agg
from pyinterpolate.variogram import TheoreticalVariogram

# Set logging
datenow = datetime.now().strftime('%Y%m%d_%H%M')
LOGGING_FILE = f'logs/analyze_area_to_area_pk_log_{datenow}.log'
LOGGING_LEVEL = 'DEBUG'
LOGGING_FORMAT = "[%(asctime)s, %(levelname)s] %(message)s"
logging.basicConfig(filename=LOGGING_FILE,
                    level=LOGGING_LEVEL,
                    format=LOGGING_FORMAT)

DATASET = '../samples/regularization/cancer_data.gpkg'
VARIOGRAM_MODEL_FILE = '../../../tests/samples/regularization/regularized_variogram.json'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 8


def select_unknown_blocks_and_ps(areal_input, point_support, block_id):
    ar_x = areal_input.cx
    ar_y = areal_input.cy
    ar_val = areal_input.value_column_name
    ps_val = point_support.value_column
    ps_x = point_support.x_col
    ps_y = point_support.y_col
    idx_col = areal_input.index_column_name

    areal_input = areal_input.data.copy()
    point_support = point_support.point_support.copy()

    sample_key = np.random.choice(list(point_support[block_id].unique()))

    unkn_ps = point_support[point_support[block_id] == sample_key][[ps_x, ps_y, ps_val]].values
    known_poses = point_support[point_support[block_id] != sample_key]
    known_poses.rename(columns={
        ps_x: 'x', ps_y: 'y', ps_val: 'ds', idx_col: 'index'
    }, inplace=True)

    unkn_area = areal_input[areal_input[block_id] == sample_key][[idx_col, ar_x, ar_y, ar_val]].values
    known_areas = areal_input[areal_input[block_id] != sample_key]
    known_areas.rename(columns={
        ar_x: 'centroid.x', ar_y: 'centroid.y', ar_val: 'ds', idx_col: 'index'
    }, inplace=True)

    return known_areas, known_poses, unkn_area, unkn_ps


def area_to_area_pk_cov(semivariogram_model: TheoreticalVariogram,
                        blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                        point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                        unknown_block: np.ndarray,
                        unknown_block_point_support: np.ndarray,
                        number_of_neighbors: int,
                        raise_when_negative_prediction=True,
                        raise_when_negative_error=True,
                        log_process=True):
    """
    Function predicts areal value in a unknown location based on the area-to-area Poisson Kriging

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

    """
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
        max_range=semivariogram_model.rang
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

    # TODO test - come back here when covariance terms are used
    # TODO - probably it should be without subtraction - those are semivariance terms, not covariances.
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


AREAL_INPUT = Blocks()
AREAL_INPUT.from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)
POINT_SUPPORT_INPUT = PointSupport()
POINT_SUPPORT_INPUT.from_files(point_support_data_file=DATASET,
                               blocks_file=DATASET,
                               point_support_geometry_col=GEOMETRY_COL,
                               point_support_val_col=POP10,
                               blocks_geometry_col=GEOMETRY_COL,
                               blocks_index_col=POLYGON_ID,
                               use_point_support_crs=True,
                               point_support_layer_name=POPULATION_LAYER,
                               blocks_layer_name=POLYGON_LAYER)

THEORETICAL_VARIOGRAM = TheoreticalVariogram()
THEORETICAL_VARIOGRAM.from_json(VARIOGRAM_MODEL_FILE)

if __name__ == '__main__':

    areas = []

    errs_cov = []
    errs_sem = []
    for _ in range(200):

        AREAL_INP, PS_INP, UNKN_AREA, UNKN_PS = select_unknown_blocks_and_ps(AREAL_INPUT,
                                                                             POINT_SUPPORT_INPUT,
                                                                             POLYGON_ID)

        if UNKN_AREA[0][0] in areas:
            continue
        else:
            areas.append(UNKN_AREA[0][0])

            uar = UNKN_AREA[0][:-1]
            uvar = UNKN_AREA[0][-1]

            pk_output_base = area_to_area_pk(semivariogram_model=THEORETICAL_VARIOGRAM,
                                             blocks=AREAL_INP,
                                             point_support=PS_INP,
                                             unknown_block=uar,
                                             unknown_block_point_support=UNKN_PS,
                                             number_of_neighbors=NN,
                                             raise_when_negative_error=False,
                                             raise_when_negative_prediction=False,
                                             log_process=True)

            pk_output_cov = area_to_area_pk_cov(semivariogram_model=THEORETICAL_VARIOGRAM,
                                                blocks=AREAL_INP,
                                                point_support=PS_INP,
                                                unknown_block=uar,
                                                unknown_block_point_support=UNKN_PS,
                                                number_of_neighbors=NN,
                                                raise_when_negative_error=False,
                                                raise_when_negative_prediction=False,
                                                log_process=True)


            errs_cov.append(pk_output_cov[1] - uvar)
            errs_sem.append(pk_output_base[1] - uvar)

    print('Semivariance errors:')
    print('Mean:', np.mean(errs_sem))
    print('STD:', np.std(errs_sem))
    print('Covariance errors:')
    print('Mean:', np.mean(errs_cov))
    print('STD:', np.std(errs_cov))
