import unittest

from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.variogram.regularization.deconvolution import Deconvolution


OUTPUT = 'samples/regularization/regularized_variogram_regular_blocks.json'
BLOCKS = 'samples/areal_data/regular_grid.geojson'
PS = 'samples/areal_data/regular_ps.geojson'
POP = 'pop'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'id'
POLYGON_VALUE = 'rate'
MAX_RANGE = 150_000
STEP_SIZE = 30_000
MAX_ITERS = 10

AREAL_INPUT = Blocks()
AREAL_INPUT.from_file(BLOCKS, value_col=POLYGON_VALUE, index_col=POLYGON_ID)
POINT_SUPPORT_INPUT = PointSupport()
POINT_SUPPORT_INPUT.from_files(point_support_data_file=PS,
                               blocks_file=BLOCKS,
                               point_support_geometry_col=GEOMETRY_COL,
                               point_support_val_col=POP,
                               blocks_geometry_col=GEOMETRY_COL,
                               blocks_index_col=POLYGON_ID,
                               use_point_support_crs=True)


class TestDeconvolution(unittest.TestCase):

    def test_fit_method(self):
        dcv = Deconvolution(verbose=False)
        dcv.fit(agg_dataset=AREAL_INPUT,
                point_support_dataset=POINT_SUPPORT_INPUT,
                agg_step_size=STEP_SIZE,
                agg_max_range=MAX_RANGE,
                model_types='all')

        fitted = dcv.initial_regularized_variogram
        initial_deviation = dcv.initial_deviation

        self.assertTrue(fitted is not None)

        expected_deviation = 0.4398
        self.assertAlmostEqual(initial_deviation, expected_deviation, 4)

    def test_transform(self):
        dcv = Deconvolution(verbose=False)

        self.assertTrue(dcv.initial_theoretical_agg_model is None)

        dcv.fit(agg_dataset=AREAL_INPUT,
                point_support_dataset=POINT_SUPPORT_INPUT,
                agg_step_size=STEP_SIZE,
                agg_max_range=MAX_RANGE,
                variogram_weighting_method='closest',
                model_types='all')

        self.assertTrue(dcv.initial_theoretical_agg_model is not None)

        dcv.transform(max_iters=MAX_ITERS)

        self.assertTrue(dcv.final_theoretical_model is not None)
        self.assertEqual(len(dcv.deviations), 11)

        try:
            with open(OUTPUT, 'r') as _:
                pass
        except FileNotFoundError:
            dcv.export_model(OUTPUT)
