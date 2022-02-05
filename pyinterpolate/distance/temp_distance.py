"""
Temporary valid distance calculation module
"""
from scipy.spatial.distance import cdist


# TEMPORARY FUNCTIONS
def temp_calc_point_to_point_distance(points_a, points_b=None):
    """temporary function for pt to pt distance estimation"""

    if points_b is None:
        distances = cdist(points_a, points_a, 'euclidean')
    else:
        distances = cdist(points_a, points_b, 'euclidean')
    return distances