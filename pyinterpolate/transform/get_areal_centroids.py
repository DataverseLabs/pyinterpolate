def get_centroids(polygon):
    """
    Function gets position x and y of polygon centroid.

    INPUT:

    :param polygon: (shapely Polygon).

    OUTPUT:

    :return: (tuple) centroid position (x, y) for a given area.
    """

    centroid_position_x = polygon.centroid.x
    centroid_position_y = polygon.centroid.y

    return centroid_position_x, centroid_position_y
