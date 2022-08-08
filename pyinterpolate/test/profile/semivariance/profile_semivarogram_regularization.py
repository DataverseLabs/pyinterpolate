import cProfile

from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.variogram.regularization.deconvolution import Deconvolution

DATASET = '../samples/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
MAX_RANGE = 400000
STEP_SIZE = 20000
MAX_ITERS = 5

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

def profile_reg():
    dcv = Deconvolution(verbose=False)
    dcv.fit_transform(agg_dataset=AREAL_INPUT,
                      point_support_dataset=POINT_SUPPORT_INPUT,
                      agg_step_size=STEP_SIZE,
                      agg_max_range=MAX_RANGE,
                      variogram_weighting_method='closest',
                      max_iters=MAX_ITERS)

    return 0


if __name__ == '__main__':
    cProfile.run('profile_reg()', filename='decon_v0.3.0.profile')
