import unittest

from pyinterpolate.processing.checks import check_limits


VALUE = 10


class TestCheckLimits(unittest.TestCase):

    def test_case_1(self):
        check_limits(
            value=VALUE, lower_limit=9, upper_limit=11, exclusive_lower=True, exclusive_upper=True
        )

        self.assertTrue(1)

    def test_case_2(self):
        with self.assertRaises(ValueError):
            check_limits(VALUE, lower_limit=10, upper_limit=11, exclusive_lower=True, exclusive_upper=True)

    def test_case_3(self):
        with self.assertRaises(ValueError):
            check_limits(VALUE, lower_limit=9, upper_limit=10, exclusive_lower=True, exclusive_upper=True)

    def test_case_4(self):
        check_limits(
            value=VALUE, lower_limit=10, upper_limit=11, exclusive_lower=False, exclusive_upper=True
        )

        self.assertTrue(1)

    def test_case_5(self):
        check_limits(
            value=VALUE, lower_limit=9, upper_limit=10, exclusive_lower=True, exclusive_upper=False
        )

        self.assertTrue(1)

    def test_case_6(self):
        with self.assertRaises(ValueError):
            check_limits(VALUE, lower_limit=10, upper_limit=10, exclusive_lower=True, exclusive_upper=True)

    def test_case_7(self):
        with self.assertRaises(ValueError):
            check_limits(VALUE, lower_limit=10, upper_limit=10, exclusive_lower=False, exclusive_upper=False)
