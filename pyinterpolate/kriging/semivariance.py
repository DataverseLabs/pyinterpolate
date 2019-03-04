import numpy as np
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance, calculate_block_to_block_distance


def calculate_semivariance(points_array, lags, step_size):
    """
    Function calculates semivariance of points in n-dimensional space.
    :param points_array: numpy array of points and values (especially DEM) where
    points_array[0] = array([point_x, point_y, ..., point_n, value])
    :param lags: array of lags between points
    :param step_size: distance which should be included in the gamma parameter which enhances range of interest
    :return: semivariance: numpy array of pair of lag and semivariance values where
    semivariance[0] = array of lags
    semivariance[1] = array of lag's values
    semivariance[2] = array of number of points in each lag

    """
    smv = []
    semivariance = []

    # Calculate distance
    try:
        distance_array = calculate_distance(points_array[:, 0:-1])
    except TypeError:
        points_array = np.asarray(points_array)
        print('Given points array has been transformed into numpy array to calculate distance')
        distance_array = calculate_distance(points_array[:, 0:-1])

    # Calculate semivariances
    for h in lags:
        gammas = []
        distances_in_range = np.where(
            np.logical_and(
                np.greater_equal(distance_array, h - step_size), np.less_equal(distance_array, h + step_size)))
        for i in range(0, len(distances_in_range[0])):
            row_x = distances_in_range[0][i]
            row_x_h = distances_in_range[1][i]
            gp1 = points_array[row_x][-1]
            gp2 = points_array[row_x_h][-1]
            g = (gp1 - gp2) ** 2
            gammas.append(g)
        if len(gammas) == 0:
            gamma = 0
        else:
            gamma = np.sum(gammas) / (2 * len(gammas))
        smv.append([gamma, len(gammas)])

    # Selecting semivariances
    for i in range(len(lags)):
        if smv[i][0] > 0:
            semivariance.append([lags[i], smv[i][0], smv[i][1]])

    semivariance = np.vstack(semivariance)

    return semivariance.T


def calculate_inblock_semivariance(blocks_dict):
    """
    Function calculates semivariance of points inside a block (area).
    :param blocks_dict: block dictionary in the minimal form: {block id: 
                                                                {'coordinates': [[x0, y0, val0],
                                                                                 [x1, y1, val1],
                                                                                 [x.., y.., val..]]
                                                                }
                                                              }
    :return: updated block dictionary with new key 'semivariance' and mean variance per area
    """
    semivariance = []
    
    blocks = list(blocks_dict.keys())

    for block in blocks:
        
        points_array = np.asarray(blocks_dict[block]['coordinates'])
        
        # Calculate semivariance
        number_of_points = len(points_array)
        p_squared = number_of_points**2
        
        for point1 in points_array:
            variances = []
            for point2 in points_array:
                v = point1[-1] - point2[-1]
                v = (v**2)
                variances.append(v)
            variance = np.sum(variances) / (2 * len(variances))
            semivariance.append(variance)
        
        semivar = np.sum(semivariance) / p_squared
        
        blocks_dict[block]['semivariance'] = semivar

    return blocks_dict


def _prepare_lags(ids_list, distances_between_blocks, lags, step):
    """
    Function prepares blocks and distances - creates dict in the form:
    {area_id: {lag: [areas in a given lag (ids)]}}
    """
    
    dbb = distances_between_blocks
    
    sorted_areas = {}
    
    for area in ids_list:
        sorted_areas[area] = {}
        for lag in lags:
            sorted_areas[area][lag] = []
            for nb in ids_list:
                if (dbb[area][nb] > lag) and (dbb[area][nb] < lag + step):
                    sorted_areas[area][lag].append(nb)
                else:
                    pass
    return sorted_areas


def _calculate_average_semivariance_for_a_lag(sorted_areas, blocks):
    """
    Function calculates average semivariance for each lag.
    yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
    y(va, va) and y(va+h, va+h) are estimated according to the function calculate_inblock_semivariance, and h
    are estimated according to the block_to_block_distances function.
    
    INPUT:
    :param sorted_areas: dict in the form {area_id: {lag: [area_ids_within_lag]}}
    :param blocks: dict with key 'semivariance' pointing the in-block semivariance of a given area
    
    OUTPUT:
    :return: list with semivariances for each lag [[lag, semivariance], [next lag, next semivariance], ...]
    """
    
    areas_ids = list(blocks.keys())
    
    lags_ids = list(sorted_areas[areas_ids[0]].keys())
    
    semivars_and_lags = []
    
    for l_id in lags_ids:
        lag = sorted_areas[areas_ids[0]][l_id]
        semivar = 0
        for a_id in areas_ids:
            base_semivariance = blocks[a_id]['semivariance']
            neighbour_areas = sorted_areas[a_id][l_id]
            no_of_areas = len(neighbour_areas)
            if no_of_areas == 0:
                semivar += 0
            else:
                s = 1 / (no_of_areas)
                semivars_sum = 0
                for area in neighbour_areas:
                    semivars_sum += base_semivariance + blocks[area]['semivariance']
                semivars_sum = s * semivars_sum
                semivar += semivars_sum
                semivar = semivar / 2
        semivars_and_lags.append([l_id, semivar])
    return semivars_and_lags


def calculate_mean_semivariance_between_blocks(blocks, lags, step):
    """
    Function calculates average semivariance between blocks separated by a vector h according to the equation:
    yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
    y(va, va) and y(va+h, va+h) are estimated according to the function calculate_inblock_semivariance, and h
    are estimated according to the block_to_block_distances function.
    
    INPUT:
    :param blocks: dictionary with a given areas and their population quadrants,
    :param lags: list of ranges between blocks,
    :param step: step size between lags.
    
    OUTPUT:
    :return: list of [[lag, semivariance], [lag_x, semivariance_x], [..., ...]]
    """
    
    # calculate inblock semivariance
    updated_blocks = calculate_inblock_semivariance(blocks)
    
    # calculate distance between blocks
    distances_between_blocks = block_to_block_distances(updated_blocks)
    
    # prepare blocks and distances - creates dict in the form:
    # {area_id: {lag: [areas in a given lag (ids)]}}
    areas_list = list(updated_blocks.keys())
    
    sorted_areas = _prepare_lags(areas_list, distances_between_blocks, lags, step)
    
    # Now calculate semivariances for each area / lag
    
    smvs = _calculate_average_semivariance_for_a_lag(sorted_areas, updated_blocks)
    
    return smvs


#### SECTION UNDER DEVELOPMENT ####

def _calculate_semivariance_block_pair(block_a_data, block_b_data):
    a_len = len(block_a_data)
    b_len = len(block_b_data)
    pa_pah = a_len * b_len
    semivariance = []
    for point1 in block_a_data:
        variances = []
        for point2 in block_b_data:
            smv = point1[-1] - point2[-1]
            smv = smv**2
            variances.append(smv)
        variance = np.sum(variances) / (2 * len(variances))
        semivariance.append(variance)
    semivar = np.sum(semivariance) / pa_pah
    return semivar
        
            
def calculate_between_blocks_semivariances(blocks):
    """Function calculates semivariance between all pairs of blocks and updates blocks dictionary with new key:
    'block-to-block semivariance' and value as a list where [[distance to another block, semivariance], ]
    
    INPUT:
    :param blocks: dictionary with a list of all blocks,
    
    OUTPUT:
    :return: updated dictionary with new key:
    'block-to-block semivariance' and the value as a list: [[distance to another block, semivariance], ]
    """
    
    bb = blocks.copy()
    
    blocks_ids = list(bb.keys())
    
    for first_block_id in blocks_ids:
        bb[first_block_id]['block-to-block semivariance'] = []
        for second_block_id in blocks_ids:
            if first_block_id == second_block_id:
                pass
            else:
                distance = calculate_block_to_block_distance(bb[first_block_id]['coordinates'],
                                                             bb[second_block_id]['coordinates'])
                smv = _calculate_semivariance_block_pair(bb[first_block_id]['coordinates'],
                                                  bb[second_block_id]['coordinates'])
                bb[first_block_id]['block-to-block semivariance'].append([distance, smv])
                
    return bb


def calculate_general_block_to_block_semivariogram(blocks, lags, step):
    blocks_ids = list(blocks.keys())
    semivariogram = []
    for lag in lags:
        semivars = []
        for block in blocks_ids:
            v = 0
            for val in blocks[block]['block-to-block semivariance']:
                if (val[0] > lag and val[0] <= lag + step):
                    v = v + val[1]
            semivars.append(v)
        l = len(semivars)
        s = np.sum(semivars) / l
        semivariogram.append([lag, s])
    return semivariogram