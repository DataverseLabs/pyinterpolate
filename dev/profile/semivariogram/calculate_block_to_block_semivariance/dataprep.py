import os

import geopandas as gpd


my_dir = os.path.dirname(__file__)
filename = 'cancer_data.gpkg'
CANCER_DATA_FILE = os.path.join(my_dir, filename)
LAYER_NAME = 'areas'
AREA_VALUES = 'rate'
AREA_INDEX = 'FIPS'
AREA_GEOMETRY = 'geometry'
PS_LAYER_NAME = 'points'
PS_VALUES = 'POP10'
PS_GEOMETRY = 'geometry'
DS = gpd.read_file(CANCER_DATA_FILE,
                   layer=LAYER_NAME)
PS = gpd.read_file(CANCER_DATA_FILE,
                   layer=PS_LAYER_NAME)


CANCER_DATA = {
    'ds': DS,
    'index_column_name': AREA_INDEX,
    'value_column_name': AREA_VALUES,
    'geometry_column_name': AREA_GEOMETRY
}

DS_C = DS.copy()
DS_C['centroid'] = DS_C.centroid

CANCER_DATA_WITH_CENTROIDS = {
    'ds': DS_C,
    'index_column_name': AREA_INDEX,
    'value_column_name': AREA_VALUES,
    'geometry_column_name': AREA_GEOMETRY,
    'representative_points_column_name': 'centroid'
}

DS_R = DS.copy()
DS_R['rep_points'] = DS_R.representative_point()

CANCER_DATA_WITH_RAND_PTS = {
    'ds': DS_R,
    'index_column_name': AREA_INDEX,
    'value_column_name': AREA_VALUES,
    'geometry_column_name': AREA_GEOMETRY,
    'representative_points_column_name': 'rep_points'
}

POINT_SUPPORT_DATA = {
    'ps': PS,
    'value_column_name': PS_VALUES,
    'geometry_column_name': PS_GEOMETRY
}
