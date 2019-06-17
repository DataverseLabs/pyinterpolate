import numpy as np
import matplotlib.pyplot as plt


class Distances:
    """Class calculates distances for a given set of points or set of areas"""

    def __init__(self, points=None, areas=None):
        self.points = points
        self.areas = areas
        self.points_distances = None
        self.areas_distances_list = None
        self.areas_distances_dict = None

    def __str__(self):
        pass

    def show_scattergram(self):
        plt.figure()
        pass

    @staticmethod
    def _calculate(pt1, pt2):
        distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        return distance

    @staticmethod
    def _calculate_distance(points, dimension):
        """
            Function calculates euclidean distance between points in n-dimensional space. Function created for
            datasets larger than 5000 rows. (It will be re-designed for parallel processing).
            :param points: numpy array with points' coordinates where each column indices new dimension and each row is
            a new coordinate set (point)
            :param dimension: dimension of dataset (1 or more than 1)
            :return: distances_list - numpy array with euclidean distances between all pairs of points.
        """
        multiple_distances_sum = 0
        distances_list = []
        if dimension == 1:
            for val in points:
                distances_list.append(np.abs(points - val))
        else:
            for row in points:
                for col in range(0, len(row)):
                    single_dist_col = (points[:, col] - row[col]) ** 2
                    if col == 0:
                        multiple_distances_sum = single_dist_col
                    else:
                        multiple_distances_sum = multiple_distances_sum + single_dist_col
                distances_list.append(np.sqrt(multiple_distances_sum))


        return np.array(distances_list)

    def calculate_distance_between_points(self, points_array=None):
        """
            Function calculates euclidean distance between points in n-dimensional space.

            :param points_array: numpy array with points' coordinates where each column indices new dimension and each
            row is a new coordinate set (point)
            :return: distances - numpy array with euclidean distances between all pairs of points.

            IMPORTANT! If input array size has x rows (coordinates) then output array size is x(cols) by x(rows)
            and each row describes distances between coordinate from row(i) with all rows.
            The first column in row is a distance between coordinate(i) and coordinate(0),
            the second row is a distance between coordinate(i) and coordinate(1) and so on.
            """

        if not points_array:
            points_array = self.points.copy()

        try:
            number_of_cols = points_array.shape[1]
        except IndexError:
            number_of_cols = 1

        distances = self._calculate_distance(points_array, number_of_cols)
        self.points_distances = distances
        return distances

    def _calculate_block_to_block_distance(self, area_block_1, area_block_2):
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

        sum_pa_pb = 0
        distance = 0

        for a_row in area_block_1:
            for b_row in area_block_2:
                weight = a_row[-1] * b_row[-1]
                sum_pa_pb = sum_pa_pb + weight
                partial_distance = self._calculate(a_row, b_row)
                distance = distance + weight * partial_distance
        distance = distance / sum_pa_pb
        return distance

    def calculate_distance_between_blocks(self, areas=None, interpretation='dict'):
        """
            Function returns distances between all blocks passed to it.
            Function has two parameters:
            :param areas: dictionary with areas (where each area has unique ID and key 'coordinates' with coordinates
            and values in the form [x, y, val]) or 3D list where each layer represents different area
            :param interpretation: 'dict' if areas are dictionary with multiple areas or list if list of coordinates
            is given as an input
            :return distances:

            if interpretation == 'dict':
            {
                'unit 0 ID': {
                    'unit 0 ID': distance to unit 0,
                    'unit n ID': distance to unit n,
                    }
                ,
                'unit n ID': {
                    'unit 0 ID': distance to unit 0,
                    'unit n ID': distance to unit n,
                    }
                ,
                'unit z ID': {
                    'unit 0 ID': distance to unit 0,
                    'unit n ID': distance to unit n,
                    }

            }

            if interpretation == 'list':
            [
                [d(coordinate 0 to coordinate 0), d(coordinate 0 to coordinate 1), d(coordinate 0 to coordinate n)],
                [d(coordinate 1 to coordinate 0), d(coordinate 1 to coordinate 1), d(coordinate 1 to coordinate n)],
                [d(coordinate n to coordinate 0), d(coordinate n to coordinate 1), d(coordinate n to coordinate n)],
            ]

        """

        if not areas:
            areas = self.areas.copy()

        if interpretation == 'dict':
            print('Selected data: dict type')  # Inform which type of data structure has been chosen
            distance_dicts = {}
            for key_a in areas.keys():
                distance_dicts[key_a] = {}
                for key_b in areas.keys():
                    if key_a == key_b:
                        distance_dicts[key_a][key_b] = 0
                    else:
                        block_1 = areas[key_a]['coordinates']
                        block_2 = areas[key_b]['coordinates']
                        distance = self._calculate_block_to_block_distance(block_1, block_2)
                        distance_dicts[key_a][key_b] = distance
            self.areas_distances_dict = distance_dicts.copy()
            return distance_dicts

        elif interpretation == 'list':
            print('Selected data: list of value lists type')  # Inform which type of data structure has been chosen
            list_of_grouped_distances = []
            for i, layer_a in enumerate(areas):
                layer_x = []
                for j, layer_b in enumerate(areas):
                    if i == j:
                        distance = 0
                    else:
                        distance = self._calculate_block_to_block_distance(layer_a, layer_b)
                    layer_x.append(distance)
                list_of_grouped_distances.append(layer_x)
            self.areas_distances_list = np.asarray(list_of_grouped_distances)
            return list_of_grouped_distances

        else:
            print('Selected data type not available. You may choose dict or list type. Please look into a docstring.')
