from datetime import datetime
from typing import Dict, Union

import numpy as np
import logging

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from pyinterpolate.kriging.models.block.area_to_area_poisson_kriging import area_to_area_pk
from pyinterpolate import area_to_point_pk
from pyinterpolate import centroid_poisson_kriging
from pyinterpolate.kriging.models.block.weight import WeightedBlock2BlockSemivariance, weights_array, \
    WeightedBlock2PointSemivariance, add_ones
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_poisson_kriging_data, prepare_pk_known_areas, \
    get_aggregated_point_support_values, get_distances_within_unknown
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, sem_to_cov, get_areal_values_from_agg
from pyinterpolate.variogram import TheoreticalVariogram

# Set logging
datenow = datetime.now().strftime('%Y%m%d_%H%M')
LOGGING_FILE = f'../logs/analyze_area_to_area_pk_log_{datenow}.log'
LOGGING_LEVEL = 'DEBUG'
LOGGING_FORMAT = "[%(asctime)s, %(levelname)s] %(message)s"
logging.basicConfig(filename=LOGGING_FILE,
                    level=LOGGING_LEVEL,
                    format=LOGGING_FORMAT)

DATASET = '../../samples/regularization/cancer_data.gpkg'
VARIOGRAM_MODEL_FILE = '../../../../tests/samples/regularization/regularized_variogram.json'
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


def area_to_point_pk2(semivariogram_model: TheoreticalVariogram,
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
                if err_to_nan:
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
                if err_to_nan:
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

    return predicted_points


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

    rmse_ata = []
    rmse_atp = []
    rmse_centroid = []
    err_ata = []
    err_atp = []
    err_centroid = []
    diff_ata_atp = []
    diff_err_ata_atp = []

    for _ in tqdm(range(200)):

        AREAL_INP, PS_INP, UNKN_AREA, UNKN_PS = select_unknown_blocks_and_ps(AREAL_INPUT,
                                                                             POINT_SUPPORT_INPUT,
                                                                             POLYGON_ID)

        if UNKN_AREA[0][0] in areas:
            continue
        else:
            areas.append(UNKN_AREA[0][0])

            uar = UNKN_AREA[0][:-1]
            uvar = UNKN_AREA[0][-1]

            pk_output_ata = area_to_area_pk(semivariogram_model=THEORETICAL_VARIOGRAM,
                                             blocks=AREAL_INP,
                                             point_support=PS_INP,
                                             unknown_block=uar,
                                             unknown_block_point_support=UNKN_PS,
                                             number_of_neighbors=NN,
                                             raise_when_negative_error=False,
                                             raise_when_negative_prediction=False,
                                             log_process=True)

            pk_output_atp = area_to_point_pk2(semivariogram_model=THEORETICAL_VARIOGRAM,
                                             blocks=AREAL_INP,
                                             point_support=PS_INP,
                                             unknown_block=uar,
                                             unknown_block_point_support=UNKN_PS,
                                             number_of_neighbors=NN,
                                             raise_when_negative_error=False,
                                             raise_when_negative_prediction=False)

            pk_cent = centroid_poisson_kriging(semivariogram_model=THEORETICAL_VARIOGRAM,
                                               blocks=AREAL_INP,
                                               point_support=PS_INP,
                                               unknown_block=uar,
                                               unknown_block_point_support=UNKN_PS,
                                               number_of_neighbors=NN,
                                               raise_when_negative_prediction=False,
                                               raise_when_negative_error=False)

            ata_pred = pk_output_ata[1]
            ata_err = pk_output_ata[2]

            atp_pred = np.sum([x[1] for x in pk_output_atp])
            atp_err = np.mean([x[2] for x in pk_output_atp])

            cent_pred = pk_cent[1]
            cent_err = pk_cent[2]

            rmse_ata.append(np.sqrt((ata_pred - uvar)**2))
            rmse_atp.append(np.sqrt((atp_pred - uvar)**2))
            rmse_centroid.append(np.sqrt((cent_pred - uvar)**2))
            err_ata.append(ata_err)
            err_atp.append(atp_err)
            err_centroid.append(cent_err)
            diff_ata_atp.append(ata_pred - atp_pred)
            diff_err_ata_atp.append(ata_err - atp_err)


    df = pd.DataFrame(
        data={
            'rmse ata': rmse_ata,
            'rmse atp': rmse_atp,
            'rmse cen': rmse_centroid,
            'err ata': err_ata,
            'err atp': err_atp,
            'err cen': err_centroid,
            'ata-atp': diff_ata_atp,
            'err ata - err atp': diff_err_ata_atp
        }
    )

    df.describe().to_csv('../test_ata_pk_data/compare_ata_atp2.csv')
