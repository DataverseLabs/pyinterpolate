# import unittest
# import numpy as np
# from pyinterpolate.distance.clusters import ClusterDetector
#
#
# POINTS = np.array(
#     [
#         [0, 0, 1],
#         [2, 2, 10],
#         [2, 0, 12],
#         [0, 2, 14]
#     ]
# )
#
#
# class TestClusterDetector(unittest.TestCase):
#
#     def test_1(self):
#         detector = ClusterDetector(ds=POINTS, verbose=False)
#         detector.fit_clusters(min_cluster_size=2)
#         labels = detector.get_labels()
#         self.assertIsInstance(labels, np.ndarray)
#         expected_labels = np.array([0, 1, 1, 0])
#         are_equal = np.array_equal(labels, expected_labels)
#         self.assertTrue(are_equal)
