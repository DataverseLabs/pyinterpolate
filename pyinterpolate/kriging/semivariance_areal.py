# Base libraries
import numpy as np
import pandas as pd

# Spatial libraries
import geopandas as gpd
from geopandas.tools import sjoin

# Custom scripts
from pyinterpolate.kriging.semivariance_base import Semivariance
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance, block_to_block_distances


class ArealSemivariance(Semivariance):

    def __init__(self, areal_data_file, areal_lags, areal_step_size,
                 population_data_file, population_value_column, population_lags, population_step_size,
                 id_column_name):
        super().__init__(areal_data_file, areal_lags, areal_step_size, id_column_name)

        self.population = population_data_file
        self.val_col = population_value_column
        self.pop_lags = population_lags
        self.pop_step = population_step_size

        self.inblock_population = None  # variable updated by blocks_semivariance() method
        self.ids = None  # variable updated by blocks_semivariance() method
        self.blocks_dict = None  # variable updated by blocks_semivariance() method


    def blocks_semivariance(self):
        """Function calculates regularized point support semivariogram in the form given in:
        Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units,
        Mathematical Geology 40(1), 101-128, 2008

        Function has the form: gamma_v(h) = gamma(v, v_h) - gamma_h(v, v) where:
        gamma_v(h) - regularized semivariogram,
        gamma(v, v_h) - semivariogram value between any two blocks separated by the distance h,
        gamma_h(v, v) - arithmetical average of within-block semivariogram

        INPUT:

        OUTPUT:
        :return: semivariance: numpy array of pair of lag and semivariance values where
                 semivariance[0] = array of lags
                 semivariance[1] = array of lag's values
                 semivariance[2] = array of number of points in each lag
        """

        within_block_semivariogram = self._calculate_mean_semivariance_between_blocks()
        return within_block_semivariogram


    def _calculate_mean_semivariance_between_blocks(self):
        """
        Function calculates average semivariance between blocks separated by a vector h according to the equation:
        yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
        y(va, va) and y(va+h, va+h) are estimated according to the function calculate_inblock_semivariance, and h
        are estimated according to the block_to_block_distances function.

        OUTPUT:
        :return: list of [[lag, semivariance], [lag_x, semivariance_x], [..., ...]]
        """

        # calculate inblock semivariance
        print('Start of the inblock semivariance calculation')
        self.inblock_semivariances = self._calculate_inblock_semivariance()
        print('Inblock semivariance calculated successfully')
        
        # calculate distance between blocks
        print('Start of the distances between blocks calculations')
        distances_between_blocks = block_to_block_distances(self.blocks_dict)
        print('Distances between blocks calculated successfully')
        
        # prepare blocks and distances - creates dict in the form:
        # {area_id: {lag: [areas in a given lag (ids)]}}
        
        print('Block to block Semivariance calculation process start...')
        print('Sorting areas by distance')
        # ranges and step from the parent class
        sorted_areas = self._prepare_lags(distances_between_blocks)
        print('Sort complete')
        
        
        # Now calculate semivariances for each area / lag
        
        smvs = self._calculate_average_semivariance_for_a_lag(sorted_areas)

        return smvs

    def _calculate_inblock_semivariance(self):
        """
        Function calculates semivariance of points inside a block (area).

        :return: dataframe with areas id and 'inblock semivariance' column as mean variance per area
        """

        areas = gpd.read_file(self.areal_data)
        population_centroids = gpd.read_file(self.population)

        # Match population centroid points with areas
        if areas.crs['init'] != population_centroids.crs['init']:
            population_centroids = population_centroids.to_crs(areas.crs)

        joined_population_points = sjoin(population_centroids, areas, how='left')
        joined_population_points = joined_population_points.dropna(axis=0)
        self.inblock_population = joined_population_points
        print('Inblock population points updated')
        
        ids = joined_population_points[self.id_field].unique()
        self.ids = list(ids)

        # Semivariance calculation and creation of dictionary with needed values
        block_dict = {}
        for single_id in ids:
            
            block_dict[single_id] = {'coordinates': 0}  # preparation of self.block_dict variable
            
            # DATA PROCESSING INTO ARRAY
            block_points = joined_population_points[joined_population_points[self.id_field] == single_id]

            points_array = super()._get_posx_posy(block_points, self.val_col, areal=False, dropna=False)
            block_dict[single_id]['coordinates'] = points_array  # update block_dict

            # Calculate semivariance
            number_of_points = len(points_array)
            p_squared = number_of_points ** 2

            distances = calculate_distance(points_array)
            try:
                semivariance = super()._calculate_semivars(self.pop_lags, self.pop_step, points_array, distances)
                semivar = np.sum(semivariance[1, :]) / p_squared
            except ValueError:
                semivar = 0
            block_dict[single_id]['inblock semivariance'] = semivar
        self.blocks_dict = block_dict

        return block_dict
    
    
    def _prepare_lags(self, distances_between_blocks):
        """
        Function prepares blocks and distances - creates dict in the form:
        {area_id: {lag: [areas in a given lag (ids)]}}
        """

        dbb = distances_between_blocks

        sorted_areas = {}

        for area in self.ids:
            sorted_areas[area] = {}
            for lag in self.ranges:
                sorted_areas[area][lag] = []
                for nb in self.ids:
                    if (dbb[area][nb] > lag) and (dbb[area][nb] < lag + self.step):
                        sorted_areas[area][lag].append(nb)
                    else:
                        pass
        return sorted_areas


    def _calculate_average_semivariance_for_a_lag(self, sorted_areas):
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

        areas_ids = self.ids

        lags_ids = list(sorted_areas[areas_ids[0]].keys())
        bb = self.blocks_dict

        semivars_and_lags = []

        for l_id in lags_ids:
            lag = sorted_areas[areas_ids[0]][l_id]
            semivar = 0
            for a_id in areas_ids:
                base_semivariance = bb[a_id]['inblock semivariance']
                neighbour_areas = sorted_areas[a_id][l_id]
                no_of_areas = len(neighbour_areas)
                if no_of_areas == 0:
                    semivar += 0
                else:
                    s = 1 / (no_of_areas)
                    semivars_sum = 0
                    for area in neighbour_areas:
                        semivars_sum += base_semivariance + bb[area]['inblock semivariance']
                    semivars_sum = s * semivars_sum
                    semivar += semivars_sum
                    semivar = semivar / 2
            semivars_and_lags.append([l_id, semivar])
        return semivars_and_lags