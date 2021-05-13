import unittest
import os
import numpy as np
import pandas as pd

from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_weighted_semivariance


class TestFitSemivariance(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFitSemivariance, self).__init__(*args, **kwargs)

        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/armstrong_data.npy')

        self.dataset = np.load(path)
        self.step_size = 1.1
        self.max_range = 10

    def test_fit_semivariance(self):

        new_col = np.arange(1, len(self.dataset) + 1)

        dataset_weights = np.zeros((self.dataset.shape[0], self.dataset.shape[1] + 1))
        dataset_weights[:, :-1] = self.dataset
        dataset_weights[:, -1] = new_col

        # Calculate weighted and non-weighted semivariance

        gamma_w = calculate_weighted_semivariance(dataset_weights, self.step_size, self.max_range)
        gamma_non = calculate_semivariance(self.dataset, self.step_size, self.max_range)

        # Fit semivariance - find optimal models
        t_non_weighted = TheoreticalSemivariogram(self.dataset, gamma_non)
        t_weighted = TheoreticalSemivariogram(dataset_weights[:, :-1], gamma_w)

        model_non_weighted = t_non_weighted.find_optimal_model(weighted=False, number_of_ranges=8)  # linear
        model_weighted = t_weighted.find_optimal_model(weighted=False, number_of_ranges=8)  # linear

        self.assertEqual(model_non_weighted, 'spherical', "Non-weighted model should be spherical")
        self.assertEqual(model_weighted, 'spherical', "Weighted model should be spherical")

    def test_fit_semivariance_io(self):
        # Prepare fake model for fit semivariance class

        fake_theoretical_smv = TheoreticalSemivariogram(None, None, False)

        nugget = 0
        sill = 20
        srange = 40
        fake_theoretical_smv.nugget = nugget
        fake_theoretical_smv.sill = sill
        fake_theoretical_smv.range = srange
        fmn = 'linear'
        fake_theoretical_smv.chosen_model_name = fmn

        my_dir = os.path.dirname(__file__)
        file_path = os.path.join(my_dir, '../sample_data/mock_model.csv')
        fake_theoretical_smv.export_model(file_path)

        # Clear model paramas and name

        fake_theoretical_smv.nugget = None
        fake_theoretical_smv.sill = None
        fake_theoretical_smv.range = None
        fake_theoretical_smv.chosen_model_name = None

        # Check if now model is not the same
        assert fake_theoretical_smv.nugget != nugget
        assert fake_theoretical_smv.range != srange
        assert fake_theoretical_smv.sill != sill
        assert fake_theoretical_smv.chosen_model_name != fmn

        # Import params
        fake_theoretical_smv.import_model(file_path)

        # Check if params are the same as at the beginning

        self.assertEqual(fake_theoretical_smv.nugget, nugget, "Problem with import/export of semivariogram nugget")
        self.assertEqual(fake_theoretical_smv.sill, sill, "Problem with import/export of semivariogram sill")
        self.assertEqual(fake_theoretical_smv.range, srange, "Problem with import/export of semivariogram range")
        self.assertEqual(fake_theoretical_smv.chosen_model_name, fmn, "Problem with import/export of semivariogram "
                                                                      "name")

    def test_semivariance_export(self):
        gamma = calculate_semivariance(self.dataset, self.step_size, self.max_range)
        theo_model = TheoreticalSemivariogram(self.dataset, gamma)
        theo_model.find_optimal_model(number_of_ranges=8)
        my_dir = os.path.dirname(__file__)
        filepath = os.path.join(my_dir, '../sample_data/test_semivariance_export.csv')
        theo_model.export_semivariance(filepath)
        df = pd.read_csv(filepath)

        columns = ['lag', 'experimental', 'theoretical']
        for c in columns:
            self.assertIn(c, df.columns, f'DataFrame is corrupted, missing {c} column')

        self.assertEqual(len(df), 10, f'DataFrame len should be 10 but it is {len(df)}')


if __name__ == '__main__':
    unittest.main()
