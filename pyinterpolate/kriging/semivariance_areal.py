# Base libraries
import numpy as np

# Spatial libraries
import pyproj
import geopandas as gpd
from geopandas.tools import sjoin

# Custom scripts
from pyinterpolate.kriging.semivariance_base import Semivariance
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance, block_to_block_distances
from pyinterpolate.kriging.helper_functions.get_centroids import get_centroids


class ArealSemivariance(Semivariance):

    def __init__(self, model,
                 areal_data_file, areal_lags, areal_step_size, data_column,
                 population_data_file, population_value_column, population_lags, population_step_size,
                 id_column_name):
        super().__init__(areal_data_file, areal_lags, areal_step_size, id_column_name)

        self.semivariance_model = model
        self.population = population_data_file
        self.val_col = population_value_column
        self.pop_lags = population_lags
        self.pop_step = population_step_size
        self.data_col = data_column

        self.inblock_population = None  # variable updated by blocks_semivariance() method
        self.areal_distances = None  # variable updated by blocks_semivariance() method
        self.ids = None  # variable updated by blocks_semivariance() method
        self.blocks_dict = None  # variable updated by blocks_semivariance() method
        self.inblock_semivariances = None  # variable updated by blocks_semivariance() method
        self.within_block_semivariogram = None
        self.semivariogram_between_blocks = None
        self.block_semivariogram = None

    def calculate_semivariances(self, distances):
        """Method predicts semivariance at a given distance. Method uses semivariance_model which is
        initialized with the class itself.

        INPUT:
        :param distances: list of distances between points.

        OUTPUT:
        :return predictions: list of semivariances for a given distances list.
        """

        predictions = self.semivariance_model.predict(distances)
        return predictions

    def blocks_semivariance(self, within_block_semivariogram=None, semivariogram_between_blocks=None):
        """Function calculates regularized point support semivariogram in the form given in:
        Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units,
        Mathematical Geology 40(1), 101-128, 2008

        Function has the form: gamma_v(h) = gamma(v, v_h) - gamma_h(v, v) where:
        gamma_v(h) - regularized semivariogram,
        gamma(v, v_h) - semivariogram value between any two blocks separated by the distance h,
        gamma_h(v, v) - arithmetical average of within-block semivariogram

        INPUT:
        :param within_block_semivariogram: mean semivariance between the blocks:
        yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
        y(va, va) and y(va+h, va+h) are the inblock semivariances of block a and block a+h separated
        by the distance h weighted by the inblock population.
        :param semivariogram_between_blocks: semivariance between all blocks calculated from the theoretical
        model.

        OUTPUT:
        :return: semivariance: numpy array of pair of lag and semivariance values where
                 semivariance[0] = array of lags
                 semivariance[1] = array of lag's values
                 semivariance[2] = array of number of points in each lag
        """
        if not within_block_semivariogram:
            within_block_semivariogram = self.calculate_mean_semivariance_between_blocks()

        if not semivariogram_between_blocks:
            semivariogram_between_blocks = self.calculate_general_block_to_block_semivariogram()

        blocks_semivar = semivariogram_between_blocks
        blocks_semivar[:, 1] = np.abs(semivariogram_between_blocks[:, 1] - within_block_semivariogram[:, 1])
        self.block_semivariogram = blocks_semivar

        return blocks_semivar

    # XXXXXXXXXXXXXXXXXXXX WITHIN-BLOCK SEMIVARIOGRAM PART XXXXXXXXXXXXXXXXXXXX

    def calculate_mean_semivariance_between_blocks(self, distances=None):
        """
        Function calculates average semivariance between blocks separated by a vector h according to the equation:
        yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
        y(va, va) and y(va+h, va+h) are estimated according to the function calculate_inblock_semivariance, and h
        are estimated according to the block_to_block_distances function.
        INPUT:
        :param distances: if given then this step of calculation is skipped

        OUTPUT:
        :return: [s, d]
                s - semivariances in the form: list of [[lag, semivariance], [lag_x, semivariance_x], [..., ...]]
                if distances:
                d - distances between blocks (dict) in the form: {area_id: {other_area_id: distance,
                                                                            other_area_id: distance,}}
                else:
                d = 0
        """

        # calculate inblock semivariance
        print('Start of the inblock semivariance calculation')
        self.inblock_semivariances = self._calculate_inblock_semivariance()
        print('Inblock semivariance calculated successfully')

        # calculate distance between blocks
        distances_between_blocks = distances
        if not distances:
            print('Start of the distances between blocks calculations')
            distances_between_blocks = block_to_block_distances(self.blocks_dict)
            self.areal_distances = distances_between_blocks
            print('Distances between blocks calculated successfully and updated')
        else:
            print('Distances between blocks given')

        # prepare blocks and distances - creates dict in the form:
        # {area_id: {lag: [areas in a given lag (ids)]}}

        print('Block to block Semivariance calculation process start...')
        print('Sorting areas by distance')
        # ranges and step from the parent class
        sorted_areas = self._prepare_lags(distances_between_blocks)
        print('Sort complete')

        # Now calculate semivariances for each area / lag

        smvs = self._calculate_average_semivariance_for_a_lag(sorted_areas)
        smvs = np.asarray(smvs)
        self.within_block_semivariogram = smvs
        return smvs

    def _calculate_inblock_semivariance(self):
        """
        Function calculates semivariance of points inside a block (area).

        :return: dataframe with areas id and 'inblock semivariance' column as mean variance per area
        """

        areas = gpd.read_file(self.areal_data)
        population_centroids = gpd.read_file(self.population)

        # Match population centroid points with areas
        if not pyproj.Proj(areas.crs).is_exact_same(pyproj.Proj(population_centroids.crs)):
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

            block_dict[single_id] = {'coordinates': 0, 'rate': 0}  # preparation of self.block_dict variable

            # DATA PROCESSING INTO ARRAY
            block_points = joined_population_points[joined_population_points[self.id_field] == single_id]

            points_array = get_centroids(block_points, self.val_col, self.id_field, areal=False, dropna=False)
            points_array = points_array[:, :-1]
            block_dict[single_id]['coordinates'] = points_array  # update block_dict
            block_dict[single_id]['rate'] = (areas[self.data_col][areas[self.id_field] == single_id]).values[0]

            # Calculate semivariance
            number_of_points = len(points_array)
            p_squared = number_of_points * number_of_points

            distances = calculate_distance(points_array)
            try:
                semivariance = self.calculate_semivariances(distances)
                try:
                    semivar = np.sum(semivariance[1, :]) / p_squared
                except IndexError:
                    semivar = np.sum(semivariance[0][0]) / p_squared
            except ValueError:
                print('Area: {}, Value error, semivariance set to 0'.format(single_id))
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

        OUTPUT:
        :return: list with semivariances for each lag [[lag, semivariance], [next lag, next semivariance], ...]
        """

        areas_ids = self.ids

        lags_ids = list(sorted_areas[areas_ids[0]].keys())
        bb = self.blocks_dict

        semivars_and_lags = []

        for l_id in lags_ids:
            semivar = 0
            for a_id in areas_ids:
                base_semivariance = bb[a_id]['inblock semivariance']
                neighbour_areas = sorted_areas[a_id][l_id]
                no_of_areas = len(neighbour_areas)
                if no_of_areas == 0:
                    semivar += 0
                else:
                    s = 1 / (2 * no_of_areas)
                    semivars_sum = 0
                    for area in neighbour_areas:
                        semivars_sum += base_semivariance + bb[area]['inblock semivariance']
                    semivars_sum = s * semivars_sum
                    semivar += semivars_sum
            semivars_and_lags.append([l_id, semivar])
        return semivars_and_lags

    # XXXXXXXXXXXXXXXXXXXX BLOCK-BLOCK SEMIVARIOGRAM PART XXXXXXXXXXXXXXXXXXXX

    def _calculate_semivariance_block_pair(self, block_a_data, block_b_data):
        a_len = len(block_a_data)
        b_len = len(block_b_data)
        pa_pah = a_len * b_len
        semivariance = []
        for point1 in block_a_data:
            variances = []
            for point2 in block_b_data:
                smv = self.semivariance_model.predict(
                    np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
                )
                variances.append(smv)
            variance = np.sum(variances) / (2 * len(variances))
            semivariance.append(variance)
        semivar = np.sum(semivariance) / pa_pah
        return semivar

    def _calculate_between_blocks_semivariances(self):
        """Function calculates semivariance between all pairs of blocks and updates blocks dictionary with new key:
        'block-to-block semivariance' and value as a list where [[distance to another block, semivariance], ]

        OUTPUT:
        :return: updated dictionary with new key:
        'block-to-block semivariance' and the value as a list: [[distance to another block, semivariance], ]
        """

        bb = self.blocks_dict.copy()

        blocks_ids = list(bb.keys())

        for first_block_id in blocks_ids:
            bb[first_block_id]['block-to-block semivariance'] = []
            for second_block_id in blocks_ids:
                if first_block_id == second_block_id:
                    pass
                else:
                    distance = self.areal_distances[first_block_id][second_block_id]

                    first_block_coordinates = bb[first_block_id]['coordinates']
                    second_block_coordinates = bb[second_block_id]['coordinates']

                    smv = self._calculate_semivariance_block_pair(first_block_coordinates,
                                                                  second_block_coordinates)

                    bb[first_block_id]['block-to-block semivariance'].append([distance, smv])
        return bb

    def calculate_general_block_to_block_semivariogram(self):
        print('Calculation of semivariances between areas separated by chosen lags')
        blocks = self._calculate_between_blocks_semivariances()
        print('Semivariance between blocks for a given lags calculated')
        blocks_ids = list(blocks.keys())
        print('Calculation of the mean semivariance for a given lag')
        semivariogram = []
        for lag in self.ranges:
            semivars = []
            for block in blocks_ids:
                v = 0
                for val in blocks[block]['block-to-block semivariance']:
                    if (val[0] > lag) and (val[0] <= lag + self.step):
                        v = v + val[1]
                semivars.append(v)
            semivars_len = len(semivars)
            average = np.sum(semivars) / semivars_len
            semivariogram.append([lag, average])
        print('End of block to block semivariogram calculation')
        self.semivariogram_between_blocks = semivariogram
        return np.asarray(semivariogram)
