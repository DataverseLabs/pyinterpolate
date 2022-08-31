import unittest
import numpy as np

from pyinterpolate.kriging.point_kriging import kriging
from pyinterpolate.test.test_kriging.consts import prepare_test_data


class TestKrige(unittest.TestCase):

    def test_base_ok(self):
        u_points = [[2, 2], [3, 3]]
        dataset, theoretical = prepare_test_data()
        known_values = [dataset[18, -1], dataset[27, -1]]
        ds = np.delete(dataset, [18, 27], axis=0)

        kriged = kriging(observations=ds, theoretical_model=theoretical,
                         points=u_points, how='ok', no_neighbors=4)

        for idx, known_value in enumerate(known_values):

            output = kriged[idx][0]
            diff = abs(known_value - output)

            self.assertTrue(diff < 1,
                            msg=f'Known value {known_value} should be close to the predicted value {output}, '
                                f'and difference should be lower than 1 unit but it is {diff} units.')

    def test_base_sk(self):
        u_points = [[2, 2], [3, 3]]
        dataset, theoretical = prepare_test_data()
        pmean = np.mean(dataset[:, -1])
        known_values = [dataset[18, -1], dataset[27, -1]]
        ds = np.delete(dataset, [18, 27], axis=0)

        kriged = kriging(observations=ds, theoretical_model=theoretical,
                         points=u_points, how='sk', no_neighbors=4, sk_mean=float(pmean))

        for idx, known_value in enumerate(known_values):
            output = kriged[idx][0]
            diff = abs(known_value - output)

            self.assertTrue(diff < 1,
                            msg=f'Known value {known_value} should be close to the predicted value {output}, '
                                f'and difference should be lower than 1 unit but it is {diff} units.')

    def test_known_ok(self):
        u_points = [[2, 2], [3, 3]]
        dataset, theoretical = prepare_test_data()
        known_values = [dataset[18, -1], dataset[27, -1]]

        kriged = kriging(observations=dataset, theoretical_model=theoretical,
                         points=u_points, how='ok', no_neighbors=4)

        for idx, known_value in enumerate(known_values):
            output = kriged[idx][0]
            self.assertAlmostEqual(output, known_value, places=6)

    def test_known_sk(self):
        u_points = [[2, 2], [3, 3]]
        dataset, theoretical = prepare_test_data()
        pmean = np.mean(dataset[:, -1])
        known_values = [dataset[18, -1], dataset[27, -1]]

        kriged = kriging(observations=dataset, theoretical_model=theoretical,
                         points=u_points, how='sk', no_neighbors=4, sk_mean=float(pmean))

        for idx, known_value in enumerate(known_values):
            output = kriged[idx][0]
            self.assertAlmostEqual(output, known_value, places=6)
