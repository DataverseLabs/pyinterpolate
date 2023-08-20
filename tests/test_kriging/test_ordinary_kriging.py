import unittest
import numpy as np

from pyinterpolate import read_txt, ExperimentalVariogram, TheoreticalVariogram
from .consts import prepare_test_data, prepare_zeros_data
from pyinterpolate.kriging.models.point.ordinary_kriging import ordinary_kriging, ordinary_kriging_from_cov


dem = read_txt('samples/point_data/txt/pl_dem_epsg2180.txt')


def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1), int(frac * len(dataset)), replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


KNOWN_POINTS, UNKNOWN_POINTS = create_model_validation_sets(dem)


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

    def test_from_covariance(self):
        exp_var = ExperimentalVariogram(input_array=KNOWN_POINTS, step_size=500, max_range=20000)
        theo_var = TheoreticalVariogram()
        theo_var.autofit(exp_var, model_name='spherical')

        for _unknown_pt in UNKNOWN_POINTS:
            predicted_sem = ordinary_kriging(
                theoretical_model=theo_var,
                known_locations=KNOWN_POINTS,
                unknown_location=_unknown_pt[:-1]
            )
            predicted_cov = ordinary_kriging_from_cov(
                theoretical_model=theo_var,
                known_locations=KNOWN_POINTS,
                unknown_location=_unknown_pt[:-1],
                sill=exp_var.variance
            )

            self.assertTrue(np.allclose(predicted_cov, predicted_sem, rtol=10, equal_nan=True))
