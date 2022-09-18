import unittest
import numpy as np
from pyinterpolate.processing.select_values import select_points_within_ellipse


INPUT_ARRAY = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4],
                        [1, 0, 5], [1, 1, 6], [1, 2, 7], [1, 3, 8], [1, 4, 9],
                        [2, 0, 10], [2, 1, 11], [2, 2, 12], [2, 3, 13], [2, 4, 14],
                        [3, 0, 15], [3, 1, 16], [3, 2, 17], [3, 3, 18], [3, 4, 19],
                        [4, 0, 20], [4, 1, 21], [4, 2, 22], [4, 3, 23], [4, 4, 24]])


EXPECTED_NS = ([2, 2, 12], [4, 2, 22])  # FROM [3, 2, 17], one lag
EXPECTED_WE = ([2, 0, 10], [2, 4, 14])  # FROM [2, 2, 12], two lags
EXPECTED_NW_SE = ([[1, 4, 9]],
                  [[0, 2, 2], [2, 4, 14]],
                  [[1, 2, 7], [3, 4, 19]],
                  [[2, 2, 12], [4, 4, 24]],
                  [[3, 2, 17]])  # FROM column number 3, lag 1
EXPECTED_NE_SW = ([[0, 4, 4], [4, 0, 20]])  # FROM [2, 2], lag 2

def contains(arr1: np.array, arr2: np.array) -> bool:
    for r in arr2:
        if (arr1 == r).all():
            return True
    return False


class TestDirectionalSelection(unittest.TestCase):

    def test_NS_selection(self):
        pt = np.array([3, 2])
        selection = select_points_within_ellipse(ellipse_center=pt,
                                                 other_points=INPUT_ARRAY[:, :-1],
                                                 lag=0,
                                                 step_size=1,
                                                 theta=0,
                                                 minor_axis_size=0.01)
        output = INPUT_ARRAY[selection]
        for row in output:
            err_msg = f'Points in direction N-S from point [3, 2] were not detected! Wrong point is: {row[:-1]}'
            is_row_in_expected = contains(row, EXPECTED_NS)
            self.assertTrue(is_row_in_expected, msg=err_msg)

    def test_WE_selection(self):
        pt = np.array([2, 2])
        selection = select_points_within_ellipse(ellipse_center=pt,
                                                 other_points=INPUT_ARRAY[:, :-1],
                                                 lag=2,
                                                 step_size=1,
                                                 theta=90,
                                                 minor_axis_size=0.1)
        output = INPUT_ARRAY[selection]
        for row in output:
            err_msg = f'Points in direction W-E and two steps away from point [2, 2] were not detected!' \
                      f' Wrong point is: {row[:-1]}'
            is_row_in_expected = contains(row, EXPECTED_WE)
            self.assertTrue(is_row_in_expected, msg=err_msg)

    def test_NW_SE_selection(self):
        points = [
            [0, 3], [1, 3], [2, 3], [3, 3], [4, 3]
        ]
        selections = []
        for pt in points:
            selection = select_points_within_ellipse(ellipse_center=pt,
                                                     other_points=INPUT_ARRAY[:, :-1],
                                                     lag=1,
                                                     step_size=1,
                                                     theta=-45,
                                                     minor_axis_size=0.01)
            output = INPUT_ARRAY[selection]
            selections.append(output)
        for idx, unit_output in enumerate(selections):
            expected = EXPECTED_NW_SE[idx]
            for jdx, out in enumerate(unit_output):
                is_point_equal = np.equal(out, expected[jdx]).all()
                err_msg = f'Points in direction NW-SE and one step away from the third column were not detected!' \
                          f' Wrong point is: {out[:-1]}'
                self.assertTrue(is_point_equal, msg=err_msg)

    def test_NE_SW_selection(self):
        point = np.array([2, 2])
        selection = select_points_within_ellipse(ellipse_center=point,
                                                 other_points=INPUT_ARRAY[:, :-1],
                                                 lag=3,
                                                 step_size=1,
                                                 theta=45,
                                                 minor_axis_size=0.01)
        output = INPUT_ARRAY[selection]

        is_point_equal = np.equal(output, EXPECTED_NE_SW).all()
        err_msg = f'Points in direction NW-SE and one step away from the third column were not detected!' \
                 f' Wrong point is: {output}'
        self.assertTrue(is_point_equal, msg=err_msg)
