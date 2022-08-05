from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.variogram.regularization.aggregated import AggregatedVariogram


if __name__ == '__main__':
    DATASET = '../samples/regularization/cancer_data.gpkg'
    POLYGON_LAYER = 'areas'
    POPULATION_LAYER = 'points'
    POP10 = 'POP10'
    GEOMETRY_COL = 'geometry'
    POLYGON_ID = 'FIPS'
    POLYGON_VALUE = 'rate'
    MAX_RANGE = 400000
    STEP_SIZE = 40000

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

    agg_var = AggregatedVariogram(
        AREAL_INPUT, STEP_SIZE, MAX_RANGE, POINT_SUPPORT_INPUT, verbose=True
    )
    _ = agg_var.regularize()

    agg_var.show_semivariograms()
