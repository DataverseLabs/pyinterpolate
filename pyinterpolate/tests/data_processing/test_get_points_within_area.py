from pyinterpolate.data_processing.data_preparation.get_points_within_area import get_points_within_area


def test_get_points():

    areal_shapefile = 'sample_data/test_areas_pyinterpolate.shp'
    points_shapefile = 'sample_data/test_points_pyinterpolate.shp'

    points_val_col = 'value'
    areal_id_col = 'id'

    points_within_area = get_points_within_area(area_shapefile=areal_shapefile,
                                                points_shapefile=points_shapefile,
                                                areal_id_col_name=areal_id_col,
                                                points_val_col_name=points_val_col,
                                                dropna=True,
                                                points_geometry_col_name='geometry',
                                                nans_to_zero=True)

    assert (len(points_within_area) == 6)


if __name__ == '__main__':
    test_get_points()
