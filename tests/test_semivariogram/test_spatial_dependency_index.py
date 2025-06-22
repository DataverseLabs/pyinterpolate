import pytest
from pyinterpolate.semivariogram.theoretical.spatial_dependency_index import calculate_spatial_dependence_index


def test_1():
    nugget = 10
    sill = 100
    eratio = (nugget / sill) * 100
    ename = 'strong'

    ratio, name = calculate_spatial_dependence_index(nugget, sill)
    assert eratio == ratio
    assert ename == name


def test_raises_value_error():
    nugget = 0
    sill = 100

    with pytest.raises(ValueError):
        calculate_spatial_dependence_index(nugget, sill)
