import os
import unittest
from tempfile import TemporaryDirectory

from pyinterpolate.viz.raster import to_tiff
from .consts import prepare_test_data


DATASET, VARIOGRAM = prepare_test_data()
SIZE = 100
NEIGHBOURS = 4

class TestWriteTIFF(unittest.TestCase):

    def test_write_tiff(self):

        with TemporaryDirectory() as tmp_dir:
            fnames = to_tiff(
                data=DATASET,
                dir_path=tmp_dir,
                dim=SIZE,
                number_of_neighbors=NEIGHBOURS,
                semivariogram_model=VARIOGRAM)

            listfiles = os.listdir(tmp_dir)

            for fname in fnames:
                self.assertTrue(os.path.basename(fname) in listfiles)

            for fname in listfiles:
                self.assertTrue('tiff' in fname or 'tfw' in fname)
