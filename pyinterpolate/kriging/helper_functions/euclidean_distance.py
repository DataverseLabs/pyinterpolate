import numpy as np


def euclidean_metrics(points, dim):
    """
    Function calculates euclidean distance between points in n-dimensional space. Function created for
    datasets larger than 5000 rows. (It will be re-designed for parallel processing).
    :param points: numpy array with points' coordinates where each column indices new dimension and each row is
    a new coordinate set (point)
    :param dim: dimension of dataset (1 or more than 1)
    :return: distances_list - numpy array with euclidean distances between all pairs of points.
    """
    distances_list = []
    if dim == 1:
        for val in points:
            distances_list.append(np.abs(points - val))
    else:
        for row in points:
            for col in range(0, len(row)):
                single_dist_col = (points[:, col] - row[col])**2
                if col == 0:
                    multiple_distances_sum = single_dist_col
                else:
                    multiple_distances_sum = multiple_distances_sum + single_dist_col
            distances_list.append(np.sqrt(multiple_distances_sum))
    return np.array(distances_list)

def calculate_distance(points_array):
    """
    Function calculates euclidean distance between points in n-dimensional space.

    :param points_array: numpy array with points' coordinates where each column indices new dimension and each row is
    a new coordinate set (point)
    :return: distances - numpy array with euclidean distances between all pairs of points.
    
    IMPORTANT! If input array size has x rows (coordinates) then output array size is x(cols) by x(rows) 
    and each row describes distances between coordinate from row(i) with all rows. 
    The first column in row is a distance between coordinate(i) and coordinate(0), 
    the second row is a distance between coordinate(i) and coordinate(1) and so on.
    """

    try:
        number_of_cols = points_array.shape[1]
    except IndexError:
        number_of_cols = 1

    distances = euclidean_metrics(points_array, number_of_cols)

    return distances

def calculate_block_to_block_distance(area_block_1, area_block_2):
    """
    Function calculates distance between two blocks based on how they are divided (into a population blocks)
    :param area_block_1: set of coordinates of each population block in the form:
    [
        [coordinate x 0, coordinate y 0, value 0],
        [...],
        [coordinate x n, coordinate y n, value n]
    ]
    :param area_block_2: the same set of coordinates as area_block_1
    :return distance: function return weighted block to block distance
    
    Equation: Dist(v_a, v_b) = 1 / (SUM_to(Pa), SUM_to(Pb) n(u_s) * n(u_si)) *
    * SUM_to(Pa), SUM_to(Pb) n(u_s) * n(u_si) ||u_s - u_si||
    where:
    Pa and Pb: number of points u_s and u_si used to discretize the two units v_a and v_b
    n(u_s) - population size in the cell u_s
    """
    
    distances_list = []
    sum_pa_pb = 0
    distance = 0
    
    for a_row in area_block_1:
        for b_row in area_block_2:
            weight = a_row[-1] * b_row[-1]
            sum_pa_pb = sum_pa_pb + weight
            partial_distance = np.sqrt((a_row[0] - b_row[0])**2 + (a_row[1] - b_row[1])**2)
            distance = distance + weight * partial_distance
    distance = distance / sum_pa_pb
    return distance

def block_to_block_distances(areas, interpretation='dict'):
    """
    Function returns distances between all blocks passed to it.
    Function has two parameters:
    :param areas: dictionary with areas (where each area has unique ID and key 'coordinates' with coordinates
    and values in the form [x, y, val]) or 3D list where each layer represents different area
    :param interpretation: 'dict' if areas are dictionary with multiple areas or list if list of coordinates
    is given as an input
    :return distances:
    
    if interporetation == 'dict':
    [
        {'unit 0 ID': {
            'unit 0 ID': distance to unit 0,
            'unit n ID': distance to unit n,
            }
        },
        {'unit n ID': {
            'unit 0 ID': distance to unit 0,
            'unit n ID': distance to unit n,
            }
        },
        {'unit z ID': {
            'unit 0 ID': distance to unit 0,
            'unit n ID': distance to unit n,
            }
        }
    ]
    
    if interpretation == 'list':
    [
        [d(coordinate 0 to coordinate 0), d(coordinate 0 to coordinate 1), d(coordinate 0 to coordinate n)],
        [d(coordinate 1 to coordinate 0), d(coordinate 1 to coordinate 1), d(coordinate 1 to coordinate n)],
        [d(coordinate n to coordinate 0), d(coordinate n to coordinate 1), d(coordinate n to coordinate n)],
    ]
    
    """
    
    if interpretation == 'dict':
        print('Selected data: dict type')  # Inform which type of data structure has been chosen
        list_of_distance_dicts = []
        for key_a in areas.keys():
            unit_dict_id = key_a
            unit_dict = {key_a: {}}
            for key_b in areas.keys():
                block_1 = areas[key_a]['coordinates']
                block_2 = areas[key_b]['coordinates']
                distance = calculate_block_to_block_distance(block_1, block_2)
                unit_dict[key_a][key_b] = distance
            list_of_distance_dicts.append(unit_dict)
        return list_of_distance_dicts
    
    elif interpretation == 'list':
        print('Selected data: list of value lists type')  # Inform which type of data structure has been chosen
        list_of_grouped_distances = []
        for layer_a in areas:
            layer_x = []
            for layer_b in areas:
                distance = calculate_block_to_block_distance(layer_a, layer_b)
                layer_x.append(distance)
            list_of_grouped_distances.append(layer_x)
        return list_of_grouped_distances
        
    else:
        print('Selected data type not available. You may choose dict or list type. Please look into a docstring.')
