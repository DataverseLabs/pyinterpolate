import unittest
import numpy as np
from pyinterpolate.processing.select_values import select_points_within_ellipse


INPUT_ARRAY = np.array(
    [
        [1, 5, 0], [2, 5, 1], [3, 5, 2], [4, 5, 3], [5, 5, 4],
        [1, 4, 5], [2, 4, 6], [3, 4, 7], [4, 4, 8], [5, 4, 9],
        [1, 3, 10], [2, 3, 11], [3, 3, 12], [4, 3, 13], [5, 3, 14],
        [1, 2, 15], [2, 2, 16], [3, 2, 17], [4, 2, 18], [5, 2, 19],
        [1, 1, 20], [2, 1, 21], [3, 1, 22], [4, 1, 23], [5, 1, 24]
    ]
)


EXPECTED_NS = ([1, 3, 10], [1, 1, 20])  # FROM [1, 2, ...], one lag
EXPECTED_WE = ([1, 2, 15], [5, 2, 19])  # FROM [3, 2, ...], two lags
EXPECTED_NW_SE = ([[4, 4, 8]],
                  [[2, 5, 1], [4, 3, 13]],
                  [[2, 4, 6], [4, 2, 18]],
                  [[2, 3, 11], [4, 1, 23]],
                  [[2, 2, 16]])  # FROM column number x=3, lag 1
EXPECTED_NE_SW = ((1, 1, 20), (5, 5, 4))  # FROM [3, 3], lag 2


def contains(arr1: np.array, arr2: np.array) -> bool:
    for r in arr2:
        if (arr1 == r).all():
            return True
    return False


class TestDirectionalSelection(unittest.TestCase):

    def test_NS_selection(self):
        pt = np.array([1, 2])
        selection = select_points_within_ellipse(ellipse_center=pt,
                                                 other_points=INPUT_ARRAY[:, :-1],
                                                 lag=0,
                                                 step_size=1,
                                                 theta=90,
                                                 minor_axis_size=0.01)
        output = INPUT_ARRAY[selection]
        for row in output:
            err_msg = f'Points in direction N-S from point [3, 2] were not detected! Wrong point is: {row[:-1]}'
            is_row_in_expected = contains(row, EXPECTED_NS)
            self.assertTrue(is_row_in_expected, msg=err_msg)

    def test_WE_selection(self):
        pt = np.array([3, 2])
        selection = select_points_within_ellipse(ellipse_center=pt,
                                                 other_points=INPUT_ARRAY[:, :-1],
                                                 lag=2,
                                                 step_size=1,
                                                 theta=0,
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
                                                     theta=135,
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
        point = np.array([3, 3])
        selection = select_points_within_ellipse(ellipse_center=point,
                                                 other_points=INPUT_ARRAY[:, :-1],
                                                 lag=3,
                                                 step_size=1,
                                                 theta=45,
                                                 minor_axis_size=0.01)
        output = INPUT_ARRAY[selection]
        for pt in output:
            tpt = (pt[0], pt[1], pt[2])
            err_msg = f'Points in direction NW-SE and one step away from the third column were not detected!' \
                      f' Wrong point is: {tpt}'
            self.assertIn(tpt, EXPECTED_NE_SW)
