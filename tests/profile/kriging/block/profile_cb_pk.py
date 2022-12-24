import cProfile
import numpy as np

from pyinterpolate.kriging.models.block.centroid_based_poisson_kriging import centroid_poisson_kriging
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.variogram import TheoreticalVariogram

DATASET = '../../samples/cancer_data.gpkg'
VARIOGRAM_MODEL_FILE = '../../samples/regularized_variogram.json'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 32


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

    unkn_area = areal_input[areal_input[block_id] == sample_key][[idx_col, ar_x, ar_y]].values
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

AREAL_INP, PS_INP, UNKN_AREA, UNKN_PS = select_unknown_blocks_and_ps(AREAL_INPUT, POINT_SUPPORT_INPUT, POLYGON_ID)

THEORETICAL_VARIOGRAM = TheoreticalVariogram()
THEORETICAL_VARIOGRAM.from_json(VARIOGRAM_MODEL_FILE)


def cbpk():
    _ = centroid_poisson_kriging(semivariogram_model=THEORETICAL_VARIOGRAM,
                                 blocks=AREAL_INP,
                                 point_support=PS_INP,
                                 unknown_block=UNKN_AREA,
                                 unknown_block_point_support=UNKN_PS,
                                 number_of_neighbors=NN,
                                 raise_when_negative_prediction=False,
                                 raise_when_negative_error=False)
    return 0


if __name__ == '__main__':
    cProfile.run('cbpk()', filename='cbprofile_v0.3.0.profile')
