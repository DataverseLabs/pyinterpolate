import unittest
import numpy as np
from pyinterpolate.distance.clusters import aggregate_cluster


POINTS = np.array(
    [
        [0, 0, 1],
        [2, 2, 10],
        [2, 0, 12],
        [0, 2, 14]
    ]
)

LABELS = np.array([0, 1, 0, 0])


class TestAggregateClusters(unittest.TestCase):

    def test_1(self):
        aggregated = aggregate_cluster(
            ds=POINTS,
            labels=LABELS,
            cluster=0,
            method="mean"
        )
        self.assertIsInstance(aggregated, np.ndarray)
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(
            aggregated[1][-1], np.mean([POINTS[0][-1], POINTS[2][-1], POINTS[3][-1]])
        )

    def test_2(self):
        self.assertRaises(KeyError, aggregate_cluster, POINTS, LABELS, 0, "abc")
