from numpy import array
from pyinterpolate.distance.point import point_distance


def test_list_input():
    points = [(0, 0), (0, 1), (0, 2)]
    other = [(2, 2), (3, 3)]
    distances = point_distance(points=points, other=other)
    assert distances.shape == (3, 2)
    assert distances.size == len(points) * len(other)


def test_tuple_input():
    points = ((0, 0), (0, 1), (0, 2))
    other = ((2, 2), (3, 3))
    distances = point_distance(points=points, other=other)
    assert distances.shape == (3, 2)
    assert distances.size == len(points) * len(other)


def test_numpy_array_input():
    points = array([(0, 0), (0, 1), (0, 2)])
    other = array([(2, 2), (3, 3)])
    distances = point_distance(points=points, other=other)
    assert distances.shape == (3, 2)
    assert distances.size == len(points) * len(other)


def test_square_distances():
    points = array([(0, 0), (0, 1), (0, 2)])
    distances = point_distance(points=points, other=points)
    assert distances.shape == (3, 3)
    assert distances.size == len(points)**2
