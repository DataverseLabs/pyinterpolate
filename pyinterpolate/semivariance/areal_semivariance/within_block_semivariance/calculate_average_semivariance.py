import numpy as np
from pyinterpolate.transform.select_values_in_range import select_values_in_range


def group_distances(distances_arrays, lags, step_size):
    """
    Function groups distances between blocks by the given lags and step size between lags.
    :param distances_arrays: (arrays)
        array[0] - list of distances between blocks,
        array[1] - list of blocks id;

        array[0]: [
                    distances[id A to id A, id A to id B, id A to id N],
                    distances[id B to id A, id B to id B, id B to id N],
                    distances[id N to id A, id N to id B, id N to id N]
                ],
        array[1]: [id A, id B, id N]
    :param lags: (array) lags between values,
    :param step_size: (float) step size between lags,
    :return grouped lags: (array) in the form: [lag, [[distances_in_range_for_area_1, number_of_distances],
                                                      [distances_in_range_for_area_2, number_of_distances]]]
    """
    grouped_lags = []
    for lag in lags:
        distances_list = []
        for dist in distances_arrays[0]:
            distances_in_range = select_values_in_range(dist, lag, step_size)
            distances_list.append([distances_in_range])
        grouped_lags.append([lag, distances_list])
    return grouped_lags


def calculate_average_semivariance(between_block_distances,
                                   semivariance_within_block_points,
                                   lags, step_size):
    """
    Function calculates average within-block semivariance between blocks.

    gamma_h(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [gamma(va, va) + gamma(va_h, va_h)]


    :param between_block_distances: distances_arrays: (arrays)
        array[0] - list of distances between blocks,
        array[1] - list of blocks ids.

        array[0]: [
                    distances[id A to id A, id A to id B, id A to id N],
                    distances[id B to id A, id B to id B, id B to id N],
                    distances[id N to id A, id N to id B, id N to id N]
                ],
        array[1]: [id A, id B, id N]
    :param semivariance_within_block_points: (numpy array) [area_id, within block semivariance]
    :param lags: (array) lags between values,
    :param step_size: (float) step size between lags,
    :return average_semivariance: (array) [lag, average semivariance]
    """
    grouped_distances = group_distances(between_block_distances, lags, step_size)
    areas_list = between_block_distances[1]

    # Select distances per lag

    avg_semivars = []
    for distance_lag in grouped_distances:
        avg_sem = []
        for idx, base_area_id in enumerate(areas_list):
            # Select internal semivariance of base area for chosen lag
            base_inblock_semivariance = semivariance_within_block_points[
                semivariance_within_block_points[:, 0] == base_area_id
            ]
            try:
                base_inblock_semivariance = base_inblock_semivariance[0][1]
            except IndexError:
                print(base_inblock_semivariance)

            if not isinstance(base_inblock_semivariance, float):
                base_inblock_semivariance = float(base_inblock_semivariance)

            # Check all distances in search radius
            neighbours_list = distance_lag[1][idx][0][0]
            no_of_areas = len(neighbours_list)
            # Skip if no neighbours
            if no_of_areas == 0:
                pass
            else:
                # Check if one neighbor
                # If so, check if this is the same area as base area
                # Calculate average semivariance from given area

                for neighbour in neighbours_list:
                    n_id = between_block_distances[1][neighbour]
                    n_semivar = semivariance_within_block_points[semivariance_within_block_points[:, 0] == n_id].copy()
                    n_semivar = n_semivar[0][1]

                    # Check type
                    if not isinstance(n_semivar, float):
                        n_semivar = float(n_semivar)

                    avg_sem.append(n_semivar + base_inblock_semivariance)

        if len(avg_sem) == 0:
            avg_sem = 0
        else:
            avg_sem = 0.5 * np.mean(avg_sem)

        # Append average semivariance for given lag
        avg_semivars.append([distance_lag[0], avg_sem])

    return np.array(avg_semivars)
