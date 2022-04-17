import unittest
import numpy as np

from pyinterpolate.kriging.kriging import kriging
from pyinterpolate.test.test_kriging.consts import prepare_test_data


class TestKrige(unittest.TestCase):

    def test_base_ok(self):
        u_points = [[2, 2], [3, 3]]
        dataset, theoretical = prepare_test_data()
        known_values = [dataset[18, -1], dataset[27, -1]]
        ds = np.delete(dataset, [18, 27], axis=0)

        kriged = kriging(observations=ds, theoretical_model=theoretical,
                         points=u_points, how='ok', min_no_neighbors=4)

        print(kriged)
        print(known_values)
