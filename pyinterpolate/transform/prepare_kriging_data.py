import numpy as np
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance,\
    calc_block_to_block_distance
from pyinterpolate.transform.set_areal_weights import get_total_value_of_area


def prepare_kriging_data(unknown_position,
                         data_array,
                         neighbors_range,
                         min_number_of_neighbors=1,
                         max_number_of_neighbors=256):
    """
    Function prepares data for kriging - array of point position, value and distance to an unknown point.

    INPUT:

    :param unknown_position: (numpy array) position of unknown value,
    :param data_array: (numpy array) known positions and their values,
    :param neighbors_range: (float) range within neighbors are included in the prediction,
    :param min_number_of_neighbors: (int) number of the n-closest neighbors used for interpolation if not any neighbor is
        selected within neighbors_range,
    :param max_number_of_neighbors: (int) maximum number of n-closest neighbors used for interpolation if there are too
        many neighbors in range. It speeds up calculations for large datasets.

    OUTPUT:
    :return: (numpy array) dataset with position, value and distance to the unknown point:
        [[x, y, value, distance to unknown position], [...]]
    """

    # Distances to unknown point
    r = np.array([unknown_position])

    known_pos = data_array[:, :-1]
    dists = calc_point_to_point_distance(r, known_pos)

    # Prepare data for kriging
    neighbors_and_dists = np.c_[data_array, dists.T]
    prepared_data = neighbors_and_dists[neighbors_and_dists[:, -1] <= neighbors_range, :]

    len_prep = len(prepared_data)

    if len_prep == 0:
        # Sort data
        sorted_neighbors_and_dists = neighbors_and_dists[neighbors_and_dists[:, -1].argsort()]
        prepared_data = sorted_neighbors_and_dists[:min_number_of_neighbors]
    elif len_prep > max_number_of_neighbors:
        sorted_neighbors_and_dists = neighbors_and_dists[neighbors_and_dists[:, -1].argsort()]
        prepared_data = sorted_neighbors_and_dists[:max_number_of_neighbors]

    return prepared_data


def prepare_poisson_kriging_data(unknown_area, points_within_unknown_area,
                                 known_areas, points_within_known_areas,
                                 number_of_neighbours, max_search_radius,
                                 weighted=False):
    """
    Function prepares data for centroid based Poisson Kriging.

    INPUT:

    :param unknown_area: (numpy array) unknown area in the form:
        [area_id, polygon, centroid x, centroid y],
    :param points_within_unknown_area: (numpy array) points and their values within the given area:
        [area_id, [point_position_x, point_position_y, value]],
    :param known_areas: (numpy array) known areas in the form:
        [area_id, polygon, centroid x, centroid y, aggregated value],
    :param points_within_known_areas: (numpy array) points and their values within the given area:
        [area_id, [point_position_x, point_position_y, value]],
    :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
    :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
        smaller than number_of_neighbours parameter then additional neighbours are included up to number of neighbors),
    :param weighted: (bool) distances weighted by population (True) or not (False).

    OUTPUT:

    :return: (numpy array) distances from known locations to the unknown location:
        [id_known, coordinate x, coordinate y, value, distance to unknown, aggregated points values within an area].
    """

    # Prepare data
    cx_cy = unknown_area[2:-1]
    r = np.array(cx_cy)

    known_centroids = known_areas.copy()
    kc_ids = known_centroids[:, 0]
    kc_vals = known_centroids[:, -1]
    kc_pos = known_centroids[:, 2:-1]

    # Build set for Poisson Kriging

    if weighted:
        known_areas_pts = points_within_known_areas.copy()

        dists = []  # [id_known, dist]

        for pt in known_areas_pts:
            d = calc_block_to_block_distance([pt, points_within_unknown_area])
            dists.append([d[0][0][1]])
        s = np.ravel(np.array(dists)).T
        kriging_data = np.c_[kc_ids, kc_pos, kc_vals, s]  # [id, coo_x, coo_y, val, dist_to_unkn]
    else:
        dists = calc_point_to_point_distance(kc_pos, [r])
        dists = dists.ravel()
        s = dists.T
        kriging_data = np.c_[kc_ids, kc_pos, kc_vals, s]  # [id, coo_x, coo_y, val, dist_to_unkn]

    # sort by distance
    kriging_data = kriging_data[kriging_data[:, -1].argsort()]

    # Get distances in max search radius
    max_search_pos = np.argmax(kriging_data[:, -1] > max_search_radius)
    output_data = kriging_data[:max_search_pos]

    # check number of observations

    if len(output_data) < number_of_neighbours:
        output_data = kriging_data[:number_of_neighbours]

    # get total points' value in each id from prepared datasets and append it to the array

    points_vals = []
    for rec in output_data:
        areal_id = rec[0]
        points_in_area = points_within_known_areas[points_within_known_areas[:, 0] == areal_id]
        total_val = get_total_value_of_area(points_in_area[0])
        points_vals.append(total_val)

    output_data = np.c_[output_data, np.array(points_vals)]
    return output_data


def _merge_vals_and_distances(known_vals, unknown_vals, distances_array):
    """
    Function prepares array of point values and respective distances for Poisson Kriging distance
    :param known_vals: (numpy array) list of known area point values - number of rows of output array,
    :param unknown_vals: (numpy array) list of unknown area point values - number of columns of output array,
    :param distances_array: (numpy array) distances array with the same number of rows as known_vals and
        the same number of columns as unknown_vals arrays,
    :return output_arr: (numpy array) array of [known point value, unknown point value, distance between points]
    """
    output = []
    for k_idx, value in enumerate(known_vals):
        output_list = [[value, x, distances_array[k_idx, u_idx]] for u_idx, x in enumerate(unknown_vals)]
        output.append(output_list)
    output_arr = np.array(output)
    return output_arr


def prepare_ata_data(points_within_unknown_area,
                     known_areas, points_within_known_areas,
                     number_of_neighbours, max_search_radius):
    """
    Function prepares data for Area to Area Poisson Kriging.

    INPUT:

    :param points_within_unknown_area: (numpy array) points and their values within the unknown area:
        [area_id, [point_position_x, point_position_y, value of point]],
    :param known_areas: (numpy array) known areas in the form:
        [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value at specific location],
    :param points_within_known_areas: (numpy array) points and their values within the given area:
        [[area_id, [point_position_x, point_position_y, value of point]], ...],
    :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
    :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
        smaller than number_of_neighbours parameter then additional neighbours are included up to number of neighbors).

    OUTPUT:

    :return output_data: (numpy array) distances from known locations to the unknown location: [id (known),
        areal value - count, [known_point_1 value, unknown_point_1 value, distance_1], total point value].
    """

    # Initialize set

    kriging_areas_ids = known_areas[:, 0]
    kriging_areal_values = known_areas[:, -1]

    # Build set for Area to Area Poisson Kriging - sort areas with distance

    known_areas_pts = points_within_known_areas.copy()

    dists = []  # [id_known, dist to unknown]

    for pt in known_areas_pts:
        d = calc_block_to_block_distance([pt, points_within_unknown_area])
        dists.append([d[0][0][1]])
    s = np.ravel(np.array(dists)).T
    kriging_data = np.c_[kriging_areas_ids, kriging_areal_values, s]  # [id, areal val, dist_to_unkn]

    # sort by distance
    kriging_data = kriging_data[kriging_data[:, -1].argsort()]

    # Get distances in max search radius
    max_search_pos = np.argmax(kriging_data[:, -1] > max_search_radius)
    output_data = kriging_data[:max_search_pos]

    # check number of observations

    if len(output_data) < number_of_neighbours:
        output_data = kriging_data[:number_of_neighbours]

    # for each of prepared id prepare distances list with points' weights for semivariogram calculation

    points_vals = []
    points_in_unknown_area = points_within_unknown_area[1][:, :-1]
    vals_in_unknown_area = points_within_unknown_area[1][:, -1]
    for rec in output_data:
        areal_id = rec[0]
        areal_value = rec[1]
        known_area = points_within_known_areas[points_within_known_areas[:, 0] == areal_id]
        known_area = known_area[0]
        points_in_known_area = known_area[1][:, :-1]
        vals_in_known_area = known_area[1][:, -1]
        distances_array = calc_point_to_point_distance(points_in_known_area, points_in_unknown_area)
        merged = _merge_vals_and_distances(vals_in_known_area, vals_in_unknown_area, distances_array)
        total_val = np.sum(known_area[1][:, 2])

        # generated_array =
        # [id, value, [known point value,
        #              unknown point value,
        #              distance between points],
        #              total point value]

        generated_array = [areal_id, areal_value, merged, total_val]
        points_vals.append(generated_array)

    output_data = np.array(points_vals)
    return output_data


def prepare_ata_known_areas(list_of_points_of_known_areas):
    """
    Function prepares known areas data for prediction.

    INPUT:

    :param list_of_points_of_known_areas: (numpy array) list of all areas' points and their values used for the
        prediction.

    OUTPUT:

    :return: (numpy array) list of arrays with areas and distances between them:
        [id base, [id other, [base point value, other point value,  distance between points]]].
    """
    all_distances_list = []
    for pt1 in list_of_points_of_known_areas:

        id_base = pt1[0][0]
        list_of_distances_from_base = [id_base, []]

        points_in_base_area = pt1[0][1][:, :-1]
        vals_in_base_area = pt1[0][1][:, -1]

        for pt2 in list_of_points_of_known_areas:

            id_other = pt2[0][0]
            points_in_other_area = pt2[0][1][:, :-1]
            vals_in_other_area = pt2[0][1][:, -1]

            distances_array = calc_point_to_point_distance(points_in_base_area, points_in_other_area)
            merged = _merge_vals_and_distances(vals_in_base_area, vals_in_other_area, distances_array)

            list_of_distances_from_base[1].append([id_other, merged])
        all_distances_list.append(list_of_distances_from_base)

    return np.array(all_distances_list)


def prepare_distances_list_unknown_area(unknown_area_points):
    """
    Function prepares distances list of unknown (single) area.

    INPUT:

    :param unknown_area_points: [pt x, pt y, val].

    OUTPUT:

    :return: [point value 1, point value 2,  distance between points].
    """
    dists = calc_point_to_point_distance(unknown_area_points[:, :-1])
    vals = unknown_area_points[:, -1]

    merged = _merge_vals_and_distances(vals, vals, dists)
    return np.array(merged)


def _merge_point_val_and_distances(unknown_point_val, known_vals, distances_array):
    """
    Function prepares array of point values and respective distances for Poisson Kriging distance.

    INPUT:

    :param unknown_point_val: (float) unknown point value,
    :param known_vals: (numpy array) list of unknown area point values,
    :param distances_array: (numpy array) distances from unknown area point to known area points.

    OUTPUT:

    :return output_arr: (numpy array) array of [unknown point value, [known points values, distance between points]].
    """
    distances_array = distances_array[:, 0]
    otp = np.array([list(x) for x in zip(known_vals, distances_array)])

    output_arr = np.array([unknown_point_val, otp])
    return output_arr


def prepare_atp_data(points_within_unknown_area,
                     known_areas, points_within_known_areas,
                     number_of_neighbours, max_search_radius):
    """
    Function prepares data for Area to Point Poisson Kriging.

    INPUT:

    :param points_within_unknown_area: (numpy array) points and their values within the given area:
        [area_id, [point_position_x, point_position_y, value of point]],
    :param known_areas: (numpy array) known areas in the form:
        [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value at specific location],
    :param points_within_known_areas: (numpy array) points and their values within the given area:
        [[area_id, [point_position_x, point_position_y, value of point]], ...],
    :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
    :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
        smaller than number_of_neighbours parameter then additional neighbours are included up to number of neighbors).

    OUTPUT:

    :return output_data: (numpy array) distances from known locations to the unknown location:
        [
            id_known,
            areal value - count,
            [
                unknown point value,
                [known point values, distance],
            total point value count
            ],
            [array of unknown area points coordinates]
        ]
    """

    # Initialize set

    kriging_areas_ids = known_areas[:, 0]
    kriging_areal_values = known_areas[:, -1]

    # Build set for Area to Area Poisson Kriging - sort areas with distance

    known_areas_pts = points_within_known_areas.copy()

    dists = []  # [id_known, dist to unknown]

    for pt in known_areas_pts:
        d = calc_block_to_block_distance([pt, points_within_unknown_area])
        dists.append([d[0][0][1]])
    s = np.ravel(np.array(dists)).T
    kriging_data = np.c_[kriging_areas_ids, kriging_areal_values, s]  # [id, areal val, dist_to_unkn]

    # sort by distance
    kriging_data = kriging_data[kriging_data[:, -1].argsort()]

    # Get distances in max search radius
    max_search_pos = np.argmax(kriging_data[:, -1] > max_search_radius)
    output_data = kriging_data[:max_search_pos]

    # check number of observations

    if len(output_data) < number_of_neighbours:
        output_data = kriging_data[:number_of_neighbours]

    # for each of prepared id prepare distances list with points' weights for semivariogram calculation

    points_vals = []
    points_and_vals_in_unknown_area = points_within_unknown_area[1]
    for rec in output_data:
        areal_id = rec[0]
        areal_value = rec[1]
        known_area = points_within_known_areas[points_within_known_areas[:, 0] == areal_id]
        known_area = known_area[0]
        points_in_known_area = known_area[1][:, :-1]
        vals_in_known_area = known_area[1][:, -1]

        # Set distances array from each point of unknown area
        merged_points_array = []
        for u_point in points_and_vals_in_unknown_area:
            u_point_dists = calc_point_to_point_distance(points_in_known_area, [u_point[:-1]])
            u_point_val = u_point[-1]
            merged = _merge_point_val_and_distances(u_point_val, vals_in_known_area, u_point_dists)
            merged_points_array.append(merged)

        total_val = np.sum(known_area[1][:, 2])

        # generated_array =
        # [[id, value, [
        # [unknown point value,
        #     [known points values,
        #      distances between points]],
        # ...],
        #  total known points value],
        # [list of uknown point coords]]

        generated_array = [areal_id, areal_value, merged_points_array, total_val]
        points_vals.append(generated_array)

    output_data = np.array(points_vals)
    return [output_data, points_within_unknown_area[1][:, :-1]]
