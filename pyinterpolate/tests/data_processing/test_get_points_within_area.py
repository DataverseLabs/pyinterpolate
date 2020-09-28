from pyinterpolate.data_processing import get_points_within_area
from sample_data.data import Data


def test_get_points():
    # Read the data

    data_class = Data()
    areal = data_class.poland_areas_dataset
    points = data_class.poland_population_dataset

    pts = get_points_within_area(areal, points,
                                 areal_id_col_name='IDx', points_val_col_name='TOT')
    assert (len(pts) == 41)


if __name__ == '__main__':
    test_get_points()
