import unittest
from typing import List

import numpy as np

from pyinterpolate.processing.transform.statistics import select_variogram_thresholds


DS = [1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 1, 1, 2, 3]
EXDS = [1.8, 3, 5.4, 7, 8]

DS0 = np.zeros(10)
DS1 = np.ones(10)
RND = np.random.random(10)
N = 5


class TestSelectVariogramThresholds(unittest.TestCase):

    def test_1(self):
        qs = select_variogram_thresholds(DS, N)
        self.assertIsInstance(qs, List)
        for idx, rc in enumerate(qs):
            self.assertAlmostEqual(rc, EXDS[idx])

    def test_2(self):
        qs = select_variogram_thresholds(DS0, N)
        self.assertEqual(np.mean(qs), 0)

    def test_3(self):
        qs = select_variogram_thresholds(DS1, N)
        self.assertEqual(np.mean(qs), 1)

    def test_4(self):
        _ = select_variogram_thresholds(RND, n_thresh=N)
        self.assertTrue(1)
