import unittest
import os
import numpy as np
from pyinterpolate.data_processing.data_preparation.read_data import read_point_data
from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_weighted_semivariance


class TestFitSemivariance(unittest.TestCase):

    def test_fit_semivariance(self):
        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, 'sample_data/poland_dem_gorzow_wielkopolski')

        dataset = read_point_data(path, 'txt')
        new_col = np.arange(1, len(dataset) + 1)

        dataset_weights = np.zeros((dataset.shape[0], dataset.shape[1] + 1))
        dataset_weights[:, :-1] = dataset
        dataset_weights[:, -1] = new_col

        # Set semivariance params
        distances = calc_point_to_point_distance(dataset_weights[:, :-2])

        maximum_range = np.max(distances)
        number_of_divisions = 10
        step_size = maximum_range / number_of_divisions
        lags = np.arange(0, maximum_range, step_size)

        # Calculate weighted and non-weighted semivariance

        gamma_w = calculate_weighted_semivariance(dataset_weights, lags, step_size)
        gamma_non = calculate_semivariance(dataset, lags, step_size)

        # Fit semivariance - find optimal models
        t_non_weighted = TheoreticalSemivariogram(dataset, gamma_non)
        t_weighted = TheoreticalSemivariogram(dataset_weights[:, :-1], gamma_w)

        model_non_weighted = t_non_weighted.find_optimal_model(weighted=False, number_of_ranges=20)
        model_weighted = t_weighted.find_optimal_model(weighted=False, number_of_ranges=20)

        self.assertEqual(model_non_weighted, 'exponential', "Non-weighted model should be exponential")
        self.assertEqual(model_weighted, 'exponential', "Weighted model should be exponential")

    def test_fit_semivariance_io(self):
        # Prepare fake model for fit semivariance class

        fake_theoretical_smv = TheoreticalSemivariogram(None, None, False)

        parameters = [0, 20, 40]
        fake_theoretical_smv.params = parameters
        fmn = 'linear'
        fake_theoretical_smv.chosen_model_name = fmn

        my_dir = os.path.dirname(__file__)
        file_path = os.path.join(my_dir, 'sample_data/mock_model.csv')
        fake_theoretical_smv.export_model(file_path)

        # Clear model paramas and name

        fake_theoretical_smv.params = [None, None, None]
        fake_theoretical_smv.chosen_model_name = None

        # Check if now model is not the same
        assert fake_theoretical_smv.params != parameters
        assert fake_theoretical_smv.chosen_model_name != fmn

        # Import params
        fake_theoretical_smv.import_model(file_path)

        # Check if params are the same as at the beginning

        self.assertEqual(fake_theoretical_smv.params, parameters, "Parameters should be [0, 20, 40]")
        self.assertEqual(fake_theoretical_smv.chosen_model_name, fmn, "Model name should be linear")


if __name__ == '__main__':
    unittest.main()
