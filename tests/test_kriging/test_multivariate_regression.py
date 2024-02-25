import unittest
import numpy as np

from pyinterpolate.kriging import MultivariateRegression


class TestMultivariateRegressionClass(unittest.TestCase):

    def test_init(self):
        mr = MultivariateRegression()
        self.assertTrue(mr)

    def test_fit(self):
        number_of_columns = np.random.randint(2, 100)
        random_arr = np.random.random(size=(1000, number_of_columns))

        mr = MultivariateRegression()
        mr.fit(random_arr)
        self.assertIsInstance(mr.features, np.ndarray)
        self.assertIsInstance(mr.intercept, float)
        self.assertIsInstance(mr.coefficients, np.ndarray)

    def test_predict(self):
        ds = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4.5]
        ])

        mr = MultivariateRegression()
        mr.fit(ds)
        predictions = mr.predict(np.array([[2.5], [4]]))
        self.assertEqual(predictions[0], 3.775)
        self.assertEqual(predictions[1], 5.5)
