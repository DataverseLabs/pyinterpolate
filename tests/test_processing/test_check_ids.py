import unittest
from pyinterpolate.processing.checks import check_ids
from pyinterpolate.processing.utils.exceptions import SetDifferenceWarning


SET_1 = {'a', 'b', 'c'}
SET_2 = {'b', 'c', 'd'}
SET_3 = {'a'}
SET_4 = {1, 2, 3}


class TestIDchecks(unittest.TestCase):

    def test_case_1(self):
        self.assertWarns(SetDifferenceWarning, check_ids, SET_1, SET_2)

    def test_case_2(self):
        self.assertWarns(SetDifferenceWarning, check_ids, SET_1, SET_3)

    def test_case_3(self):
        self.assertWarns(SetDifferenceWarning, check_ids, SET_3, SET_2)

    def test_case_4(self):
        self.assertWarns(SetDifferenceWarning, check_ids, SET_4, SET_2)
