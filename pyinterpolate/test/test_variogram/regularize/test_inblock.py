import unittest

from pyinterpolate.processing.polygon.structure import get_polyset_from_file
from pyinterpolate.processing.point.structure import get_point_support_from_files
from pyinterpolate.variogram import build_experimental_variogram
from pyinterpolate.variogram.regularization.inblock_semivariance import calculate_inblock_semivariance


POLYGON_DATA = '../../samples/regularization/counties_cancer_data.json'
POPULATION_DATA = '../../samples/regularization/population_counts.json'
POINT_SUPPORT_VALUE_COLUMN = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID_COLUMN = 'FIPS'
POLYGON_VALUE_COLUMN = 'rate'

AREAL_INPUT = get_polyset_from_file(POLYGON_DATA, value_col=POLYGON_VALUE_COLUMN, index_col=POLYGON_ID_COLUMN)
POINT_SUPPORT_INPUT = get_point_support_from_files(point_support_data_file=POPULATION_DATA,
                                                   polygon_file=POLYGON_DATA,
                                                   point_support_geometry_col=GEOMETRY_COL,
                                                   point_support_val_col=POINT_SUPPORT_VALUE_COLUMN,
                                                   polygon_geometry_col=GEOMETRY_COL,
                                                   polygon_index_col=POLYGON_ID_COLUMN,
                                                   use_point_support_crs=True,
                                                   dropna=True)


class TestDeconvolution(unittest.TestCase):

    def test_calculate_inblock(self):
        # Variogram model
        # experimental_variogram_of_areal_data = build_experimental_variogram(AREAL_INPUT, )


        # Single core

        # inblock_semivariances = calculate_inblock_semivariance(POINT_SUPPORT_INPUT['data'], )
        self.assertTrue(1)