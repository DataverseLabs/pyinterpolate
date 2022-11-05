import unittest
import numpy as np
from .consts import prepare_test_data, prepare_zeros_data
from pyinterpolate.kriging.models.point.ordinary_kriging import ordinary_kriging


class TestOrdinaryKriging(unittest.TestCase):

    def test_base_case(self):
        dataset, theoretical = prepare_test_data()
        unknown_point = [2, 2]
        known_value = dataset[18, -1]
        ds = np.delete(dataset, 18, axis=0)

        kriged = ordinary_kriging(theoretical_model=theoretical,
                                  known_locations=ds,
                                  unknown_location=unknown_point,
                                  no_neighbors=8)

        diff = abs(known_value - kriged[0])

        self.assertTrue(diff < 1, msg=f'Known value {known_value} should be close to the predicted value {kriged[0]}, '
                                      f'and difference should be lower than 1 unit but it is {diff} units.')

    def test_zeros_case(self):
        dataset, theoretical = prepare_zeros_data()

        for _ in range(0, 10):
            unknown_point = [np.random.rand() * 7, np.random.rand() * 7]  # any point within (0-7)
            try:
                _ = ordinary_kriging(theoretical_model=theoretical,
                                     known_locations=dataset,
                                     unknown_location=unknown_point,
                                     no_neighbors=4)
            except np.linalg.LinAlgError:
                self.assertTrue(True)
            except RuntimeError:
                self.assertTrue(True)
            else:
                msg = 'Every test should raise LinAlgError | RunetimeError because there are only zeros and ' \
                      'algorithm creates the singular matrix.'
                raise ValueError(msg)

    def test_value_known_location(self):
        dataset, theoretical = prepare_test_data()
        test_point = [2, 2]  # 18
        expected_value = 18
        kriged = ordinary_kriging(theoretical_model=theoretical,
                                  known_locations=dataset,
                                  unknown_location=test_point,
                                  no_neighbors=4)

        self.assertAlmostEqual(kriged[0], expected_value, places=6)
