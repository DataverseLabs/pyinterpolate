import unittest
import numpy as np

from pyinterpolate.processing.select_values import select_points_within_triangle, generate_triangles,\
    select_points_within_ellipse


INPUT_ARRAY = np.array(
    [
        [1, 5, 0], [2, 5, 1], [3, 5, 2], [4, 5, 3], [5, 5, 4],
        [1, 4, 5], [2, 4, 6], [3, 4, 7], [4, 4, 8], [5, 4, 9],
        [1, 3, 10], [2, 3, 11], [3, 3, 12], [4, 3, 13], [5, 3, 14],
        [1, 2, 15], [2, 2, 16], [3, 2, 17], [4, 2, 18], [5, 2, 19],
        [1, 1, 20], [2, 1, 21], [3, 1, 22], [4, 1, 23], [5, 1, 24]
    ]
)


class TestSelectionAlgorithms(unittest.TestCase):

    def test_triangular_selection(self):
        # Triangle A:
        tra = (
            (2.5, 1.5), (4.5, 3), (2.5, 4.5)
        )
        expected_points_a = [
            (3, 2), (3, 3), (3, 4), (4, 3)
        ]

        sel_a = select_points_within_triangle(tra, points=INPUT_ARRAY)
        output_a = INPUT_ARRAY[sel_a]

        for coords in output_a:
            crs = (coords[0], coords[1])
            self.assertIn(crs, expected_points_a)

        # Triangle B - create one
        ptb = np.array([[2.5, 3]])
        trb = generate_triangles(ptb, 2, 90, 1.5)

        for edge in trb[0][0]:
            self.assertIn(edge, tra)

        # Triangle C
        trc = (
            (2.5, 1.5), (4.5, 3.5), (5.5, 0.5)
        )

        ptc = np.array([[3.5, 2.5]])
        trc_gen = generate_triangles(ptc, 2.84, 45, 1)

        ell_gen = select_points_within_ellipse(ptc[0], INPUT_ARRAY[:, :-1],
                                               lag=3, step_size=1.5, theta=45, minor_axis_size=0.1)
        ell_sel = INPUT_ARRAY[ell_gen]
        print(trc_gen[0][0])
