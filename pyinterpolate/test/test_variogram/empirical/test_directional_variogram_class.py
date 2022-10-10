# Core modules
import unittest
# Math modules
import numpy as np
# Tested module
from pyinterpolate.variogram.empirical.experimental_variogram import DirectionalVariogram
# Test data
from pyinterpolate.test.test_variogram.empirical.consts import EmpiricalVariogramTestData, EmpiricalVariogramClassData


gen_data = EmpiricalVariogramTestData()
cls_data = EmpiricalVariogramClassData()


class TestDirectionalVariogram(unittest.TestCase):

    pass