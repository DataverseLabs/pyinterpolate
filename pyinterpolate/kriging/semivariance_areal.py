# Base libraries
import numpy as np
import pandas as pd

# Spatial libraries
import geopandas as gpd
from geopandas.tools import sjoin

# Custom scripts
from pyinterpolate.kriging.semivariance_base import Semivariance
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance


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
        self.inblock_semivariances = None  # variable updated by blocks_semivariance() method


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
        print(within_block_semivariogram)


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
        updated_blocks = self._calculate_inblock_semivariance()

        # # calculate distance between blocks
        # distances_between_blocks = calculate_block_to_block_distance(updated_blocks)
        #
        # # prepare blocks and distances - creates dict in the form:
        # # {area_id: {lag: [areas in a given lag (ids)]}}
        # areas_list = list(updated_blocks.keys())
        #
        # sorted_areas = _prepare_lags(areas_list, distances_between_blocks, lags, step)
        #
        # # Now calculate semivariances for each area / lag
        #
        # smvs = _calculate_average_semivariance_for_a_lag(sorted_areas, updated_blocks)

        return 0

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
        joined_population_points = joined_population_points.dropna(axis=1)
        self.inblock_population = joined_population_points
        print('Inblock population points updated')

        ids = joined_population_points[self.id_field]

        # Semivariance calculation
        semivariances = []
        for single_id in ids:
            # DATA PROCESSING INTO ARRAY
            block_points = joined_population_points[joined_population_points[self.id_field] == single_id]

            points_array = super()._get_posx_posy(block_points, self.val_col, dropna=False)

            # Calculate semivariance
            number_of_points = len(points_array)
            p_squared = number_of_points ** 2

            distances = calculate_distance(points_array)
            semivariance = super()._calculate_semivars(self.pop_lags, self.pop_step, points_array, distances)

            semivar = np.sum(semivariance[1, :]) / p_squared
            semivariances.append([ids, semivar])

        df = pd.DataFrame(semivariances, columns=[super().id_field, 'inblock semivariance'])

        return df