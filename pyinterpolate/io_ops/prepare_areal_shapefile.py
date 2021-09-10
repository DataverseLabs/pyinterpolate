import numpy as np
import geopandas as gpd


def prepare_areal_shapefile(areal_file_address,
                            id_column_name=None,
                            value_column_name=None,
                            geometry_column_name='geometry',
                            dropnans=True
                            ) -> gpd.GeoDataFrame:
    """

    Function prepares areal shapefile.

    INPUT:

    :param areal_file_address: (string) path to the shapefile with area data,
    :param id_column_name: (string) id column name, if not provided then index column is treated as the id,
    :param value_column_name: (string) value column name, if not provided then all values are set to NaN,
    :param geometry_column_name: (string) default is 'geometry' as in GeoPandas GeoDataFrames,
    :param dropnans: (bool) if True then rows with NaN are dropped.

    OUTPUT:

    :return: (gpd.GeoDataFrame)
    """

    # Test if value column name is None and dropnans is True
    if (value_column_name is None) and dropnans:
        raise TypeError('You cannot leave value_column_name as None and set dropnans to True because function '
                        'will return empty frame')

    shapefile = gpd.read_file(areal_file_address)

    # First get geometry as a geoseries
    gdf = gpd.GeoDataFrame(shapefile[geometry_column_name])

    # Now add index if not the same as the base index
    if id_column_name is None:
        gdf['area.id'] = gdf.index
    else:
        gdf['area.id'] = shapefile[id_column_name]

    # Now add value column name
    if value_column_name is None:
        gdf['area.value'] = np.nan
    else:
        gdf['area.value'] = shapefile[value_column_name]

    # Now get centroids
    gdf['area.centroid'] = gdf.centroid

    # Now remove nans
    if dropnans:
        gdf.dropna(axis=0, inplace=True, how='any')

    return gdf
