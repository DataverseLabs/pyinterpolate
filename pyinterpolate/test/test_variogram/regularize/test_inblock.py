import unittest

from pyinterpolate.processing.polygon.structure import get_polyset_from_file

AREAL_DATA = '../sample_data/areal_data/cancer_data.shp'
POINT_DATA = '../sample_data/population_data/cancer_population_base.shp'


AREAL_DATA = get_polyset_from_file(SHAPEFILE, value_col='value', index_col='idx')


class TestDeconvolution(unittest.TestCase):

    def test_calculate_inblock(self):
        # Single core
        print(AREAL_DATA)
        assert(True)