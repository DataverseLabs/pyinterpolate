import numpy as np


def get_centroids(geodataframe, value_col_name, id_col_name, areal=True, dropna=False):
    """Function prepares array for distances calculation from the centroids of areal blocks
    
       INPUT:
        :param geodataframe: dataframe with spatial data - areas or set of points,
        :param value_col_name: name of the column with value which is passed as the last column of the
        output array,
        :param id_col_name: name of the column with unique ID of each region,
        :param dropna: default is False. If True then all areas (points) of unknown coordinates or values
        are dropped from the analysis. It is important to drop any NaN's before the modeling phase because
        Kriging is not able to learn from not-known values.
        
       OUTPUT:
        :return pos_and_vals: numpy array of the form [[coordinate x1, coordinate y1, value1, area_id],
                                                       [coordinate x2, coordinate y2, value2, area_id],
                                                       [...., ...., ....],]
    """
    
    gdf = geodataframe.copy()
    
    col_x = 'centroid_pos_x'
    col_y = 'centroid_pos_y'
    
    if areal:
        gdf[col_x] = gdf['geometry'].apply(lambda _: _.centroid.x)
        gdf[col_y] = gdf['geometry'].apply(lambda _: _.centroid.y)
    else:
        try:
            gdf[col_x] = gdf['geometry'].apply(lambda _: _.x)
            gdf[col_y] = gdf['geometry'].apply(lambda _: _.y)
        except AttributeError:
            gdf[col_x] = gdf['geometry'].apply(lambda _: _[0].x)
            gdf[col_y] = gdf['geometry'].apply(lambda _: _[0].y)

    columns_to_hold = [col_x, col_y, value_col_name, id_col_name]
    columns = list(gdf.columns)

    # remove rows with nan
    if dropna:
        gdf.dropna(axis=0, inplace=True)

    # remove unwanted columns
    for col in columns:
        if col not in columns_to_hold:
            gdf.drop(labels=col, axis=1, inplace=True)

    # set order of columns
    gdf = gdf[columns_to_hold]

    pos_and_vals = np.asarray(gdf.values)
    return pos_and_vals
