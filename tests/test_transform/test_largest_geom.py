from shapely.geometry import Polygon, MultiPolygon
from pyinterpolate.transform.geo import largest_geometry


def test_case_1():
    pol1 = Polygon([[0, 0], [1, 1], [1, 0], [0, 0]])
    pol2 = Polygon([[1, 2], [4, 6], [2, 0], [1, 2]])

    mpol = MultiPolygon([pol1, pol2])

    assert largest_geometry(mpol) == pol2
