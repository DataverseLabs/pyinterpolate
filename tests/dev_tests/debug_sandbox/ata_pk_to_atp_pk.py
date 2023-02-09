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
VARIOGRAM_MODEL_FILE = '../../samples/regularization/regularized_variogram.json'
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

            pk_output_atp = area_to_point_pk(semivariogram_model=THEORETICAL_VARIOGRAM,
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

    df.describe().to_csv('test_ata_pk_data/compare_ata_atp.csv')
