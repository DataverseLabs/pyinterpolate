# numerical libraries
import numpy as np

# spatial libraries
import geopandas as gpd
from geopandas.tools import sjoin

# pyinterpolate scripts
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_block_to_block_distance


class Semivariance:

    def __init__(self, areal_data_file, lags=None, step_size=None, id_field='ID'):
        """Class for calculation of Areal semivariance. It us used by the methods of Area-to-Area and Area-to-Point
           Kriging. Class has two main methods: semivariance_from_centroids and semivariance_from_areal_data.
           Class is initialized by:
           areal_data: the dictionary of areas and their values,
           population_data: vector files with population centroids derived from population blocks for each area.
           To maintain stability areas in the areal_data and population blocks in the population_data must have
           the same ID filed - id_field, which allows algorithm to merge those datasets.

           Calculation methods of the class:
           C1. centroids_semivariance(lags=self.lags, step_size=self.step_size)
           C2. blocks_semivariance(lags=self.lags, step_size=self.step_size)

           Visualization methods of the class:


           Private methods of the class and their relations to the specific class variables and methods:


        """
        self.areal_data = areal_data_file
        
        self.ranges = lags
        self.step = step_size
        
        self.id_field = id_field

        self.centroids = None  # variable updated by the centroids_semivariance() method
        self.point_support_semivariance = None  # variable updated by the centroids_semivariance() method

                            ###### BASE METHODS ######

    def centroids_semivariance(self, lags=None, step_size=None, update=True, data_column='DATA'):
        """
        Function calculates semivariance of areal centroids and their values.
        :param lags: array of lags between points
        :param step_size: distance which should be included in the gamma parameter which enhances range of interest
        :param update: if True then class centroids and point_support_semivariance variables will be updated
        :return: semivariance: numpy array of pair of lag and semivariance values where
                 semivariance[0] = array of lags
                 semivariance[1] = array of lag's values
                 semivariance[2] = array of number of points in each lag
        """

        # Set lags
        if not lags:
            lags = self.ranges

        # Set step size
        if not step_size:
            step_size = self.step

        # Calculate centroids positions
        centroids = self._centroids_from_shp(data_column)
        
        # Calculate distances
        try:
            distance_array = calculate_distance(centroids[:, 0:-1])
        except TypeError:
            centroids = np.asarray(centroids)
            print('Given points array has been transformed into numpy array to calculate distance')
            distance_array = calculate_distance(centroids[:, 0:-1])
            
        semivariance = self._calculate_semivars(lags, step_size, centroids, distance_array)
        
        # Update object
        if update:
            self.centroids = centroids
            self.point_support_semivariance = semivariance

        return semivariance

                            ###### PRIVATE METHODS ######

    def _calculate_semivars(self, lags, step_size, points_array, distances_array):
        """Method calculates semivariance.

        INPUT:
        :param lags: list of lags,
        :param step_size: step between lags,
        :param points_array: array with points and their values,
        :param distances_array: array with distances between points

        OUTPUT:
        :return semivariance: numpy array of pair of lag and semivariance values where
                 semivariance[0] = array of lags
                 semivariance[1] = array of lag's values
                 semivariance[2] = array of number of points in each lag"""
        smv = []
        semivariance = []

        # Calculate semivariances
        for h in lags:
            gammas = []
            distances_in_range = np.where(
                np.logical_and(
                    np.greater_equal(distances_array, h - step_size), np.less_equal(distances_array, h + step_size)))
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

    
    def _centroids_from_shp(self, data_column):
        """Method calculates centroids of areas from the given polygon file and returns numpy array with coordinates
        and values for each centroid

        INPUT:
        :param data_column: Column with data values (usually rates)

        OUTPUT:
        :return centroids_and_vals: numpy array in the form of [[coordinate x,
                                                                 coordinate y,
                                                                 value of a given area]]"""
        gdf = gpd.read_file(self.areal_data)

        centroids_and_vals = self._get_posx_posy(gdf, data_column, dropna=True)

        return centroids_and_vals


    def _get_posx_posy(self, geo_df, value_column_name, dropna=False):
        geo_dataframe = geo_df.copy()

        col_x = 'centroid_pos_x'
        col_y = 'centroid_pos_y'

        geo_dataframe[col_x] = geo_dataframe['geometry'].apply(lambda _: _.centroid.x)
        geo_dataframe[col_y] = geo_dataframe['geometry'].apply(lambda _: _.centroid.y)

        columns_to_hold = [col_x, col_y, value_column_name]
        columns = list(geo_dataframe.columns)

        # remove rows with nan
        if dropna:
            geo_dataframe.dropna(axis=0, inplace=True)

        # remove unwanted columns
        for col in columns:
            if col not in columns_to_hold:
                geo_dataframe.drop(labels=col, axis=1, inplace=True)

        # set order of columns
        geo_dataframe = geo_dataframe[columns_to_hold]

        pos_and_vals = np.asarray(geo_dataframe.values)
        return pos_and_vals

#
#
#
# def _prepare_lags(ids_list, distances_between_blocks, lags, step):
#     """
#     Function prepares blocks and distances - creates dict in the form:
#     {area_id: {lag: [areas in a given lag (ids)]}}
#     """
#
#     dbb = distances_between_blocks
#
#     sorted_areas = {}
#
#     for area in ids_list:
#         sorted_areas[area] = {}
#         for lag in lags:
#             sorted_areas[area][lag] = []
#             for nb in ids_list:
#                 if (dbb[area][nb] > lag) and (dbb[area][nb] < lag + step):
#                     sorted_areas[area][lag].append(nb)
#                 else:
#                     pass
#     return sorted_areas
#
#
# def _calculate_average_semivariance_for_a_lag(sorted_areas, blocks):
#     """
#     Function calculates average semivariance for each lag.
#     yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
#     y(va, va) and y(va+h, va+h) are estimated according to the function calculate_inblock_semivariance, and h
#     are estimated according to the block_to_block_distances function.
#
#     INPUT:
#     :param sorted_areas: dict in the form {area_id: {lag: [area_ids_within_lag]}}
#     :param blocks: dict with key 'semivariance' pointing the in-block semivariance of a given area
#
#     OUTPUT:
#     :return: list with semivariances for each lag [[lag, semivariance], [next lag, next semivariance], ...]
#     """
#
#     areas_ids = list(blocks.keys())
#
#     lags_ids = list(sorted_areas[areas_ids[0]].keys())
#
#     semivars_and_lags = []
#
#     for l_id in lags_ids:
#         lag = sorted_areas[areas_ids[0]][l_id]
#         semivar = 0
#         for a_id in areas_ids:
#             base_semivariance = blocks[a_id]['semivariance']
#             neighbour_areas = sorted_areas[a_id][l_id]
#             no_of_areas = len(neighbour_areas)
#             if no_of_areas == 0:
#                 semivar += 0
#             else:
#                 s = 1 / (no_of_areas)
#                 semivars_sum = 0
#                 for area in neighbour_areas:
#                     semivars_sum += base_semivariance + blocks[area]['semivariance']
#                 semivars_sum = s * semivars_sum
#                 semivar += semivars_sum
#                 semivar = semivar / 2
#         semivars_and_lags.append([l_id, semivar])
#     return semivars_and_lags
#
#
#
#
#
# #### SECTION UNDER DEVELOPMENT ####
#
# def _calculate_semivariance_block_pair(block_a_data, block_b_data):
#     a_len = len(block_a_data)
#     b_len = len(block_b_data)
#     pa_pah = a_len * b_len
#     semivariance = []
#     for point1 in block_a_data:
#         variances = []
#         for point2 in block_b_data:
#             smv = point1[-1] - point2[-1]
#             smv = smv**2
#             variances.append(smv)
#         variance = np.sum(variances) / (2 * len(variances))
#         semivariance.append(variance)
#     semivar = np.sum(semivariance) / pa_pah
#     return semivar
#
#
# def calculate_between_blocks_semivariances(blocks):
#     """Function calculates semivariance between all pairs of blocks and updates blocks dictionary with new key:
#     'block-to-block semivariance' and value as a list where [[distance to another block, semivariance], ]
#
#     INPUT:
#     :param blocks: dictionary with a list of all blocks,
#
#     OUTPUT:
#     :return: updated dictionary with new key:
#     'block-to-block semivariance' and the value as a list: [[distance to another block, semivariance], ]
#     """
#
#     bb = blocks.copy()
#
#     blocks_ids = list(bb.keys())
#
#     for first_block_id in blocks_ids:
#         bb[first_block_id]['block-to-block semivariance'] = []
#         for second_block_id in blocks_ids:
#             if first_block_id == second_block_id:
#                 pass
#             else:
#                 distance = calculate_block_to_block_distance(bb[first_block_id]['coordinates'],
#                                                              bb[second_block_id]['coordinates'])
#                 smv = _calculate_semivariance_block_pair(bb[first_block_id]['coordinates'],
#                                                   bb[second_block_id]['coordinates'])
#                 bb[first_block_id]['block-to-block semivariance'].append([distance, smv])
#
#     return bb
#
#
# def calculate_general_block_to_block_semivariogram(blocks, lags, step):
#     blocks_ids = list(blocks.keys())
#     semivariogram = []
#     for lag in lags:
#         semivars = []
#         for block in blocks_ids:
#             v = 0
#             for val in blocks[block]['block-to-block semivariance']:
#                 if (val[0] > lag and val[0] <= lag + step):
#                     v = v + val[1]
#             semivars.append(v)
#         l = len(semivars)
#         s = np.sum(semivars) / l
#         semivariogram.append([lag, s])
#     return semivariogram