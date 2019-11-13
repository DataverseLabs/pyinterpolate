import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
import pyproj

from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance
from pyinterpolate.kriging.helper_functions.euclidean_distance import block_to_block_distances
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_block_to_block_distance
from pyinterpolate.kriging.helper_functions.get_centroids import get_centroids


# TODO: remove matrix structures
class PKrige:
    """
    Class for Poisson Kriging, Area-to-area (ATA) and Area-to-Point (ATP) Poisson Kriging interpolation of
    the unknown values in a given location (position). Class takes two arguments during the initialization:
    counties_data -
    population_data -
    semivariogram_model -

    Available methods:


    Static methods:


    Data visualization methods:


    Example how to prepare model:



    """

    def __init__(self, counties_data, population_data, semivariogram_model,
                 areal_id_col_name, areal_val_name, population_val_col_name):
        """
        :param counties_data: address to dataset with areal data (polygons),
        :param population_data: address to dataset with point data (centroids),
        :param semivariogram_model: deconvoluted semivariogram,
        :param areal_id_col_name: id column name of areal data,
        :param areal_val_name: value column name of areal data,
        :param population_val_col_name: value column name of population data,

        """
        self.model = semivariogram_model
        self.counties_dataset = gpd.read_file(counties_data)
        self.population_dataset = gpd.read_file(population_data)
        self.joined_datasets = self._join_datasets()
        self.id_col = areal_id_col_name
        self.val_col = areal_val_name
        self.pop_col = population_val_col_name
        self.total_population_per_unit = self._get_tot_population(self.id_col,
                                                                  self.pop_col)
        self.centroids_of_areal_data = self._get_areal_centroids(self.counties_dataset,
                                                                 self.val_col,
                                                                 self.id_col)
        self.prepared_data = None
        self.global_mean = self.counties_dataset[self.val_col].mean()  # Only for test purposes

        self.unknown_area_id = None

        # Parameters
        self.lags = None
        self.step = None
        self.max_no_of_observations = None
        self.max_search_radius = None

    # Data preparation functions

    def set_params(self,
                   lags_number,
                   lag_step_size,
                   max_no_of_observations,
                   search_radius
                   ):
        # Function sets class parameters

        self.lags = lags_number
        self.step = lag_step_size
        self.max_no_of_observations = max_no_of_observations
        self.max_search_radius = search_radius
        print('Parameters have been set')

    def prepare_data(self, unknown_area_id, weighted=False, verbose=False):
        """
        :param unknown_area_id: id of the area with unknown value,
        :param weighted: distances weighted by population (True) or not (False),
        :param verbose: if True then method informs about the successful operation.
        :return output_data: prepared dataset which contains:
        [[known_position_x, known_position_y, value, area id, distance_to_unknown_position], [...]]
        """

        # Simple Poisson Kriging (with centroid-based approach)

        self.unknown_area_id = unknown_area_id

        # Get unknown area centroid

        centroid_pos = self.centroids_of_areal_data[
            self.centroids_of_areal_data[:, -1] == unknown_area_id]

        cx_cy = centroid_pos[0][:2]

        r = np.array([cx_cy])
        known_centroids = self.centroids_of_areal_data[
            self.centroids_of_areal_data[:, -1] != unknown_area_id]
        kc = known_centroids[:, :2]

        # Build set for Poisson Kriging

        if weighted:
            weighted_distances = self._calculate_weighted_distances(unknown_area_id)

            s = []
            for wd in weighted_distances:
                for kc in known_centroids:
                    if wd[1] in kc:
                        s.append(wd[0])
                        break
                    else:
                        pass

            s = np.array(s).T

            kriging_data = np.c_[known_centroids, s]  # [coo_x, coo_y, val, id, weighted_dist_to_unkn]
        else:

            distances_array = np.zeros(kc.shape)
            for i in range(0, r.shape[1]):
                distances_array[:, i] = (kc[:, i] - r[:, i]) ** 2
            s = distances_array.sum(axis=1)
            s = np.sqrt(s)
            s = s.T
            kriging_data = np.c_[known_centroids, s]  # [coo_x, coo_y, val, id, dist_to_unkn]

        # remove nans
        kriging_data = kriging_data[~np.isnan(kriging_data).any(axis=1)]

        # sort by distance
        kriging_data = kriging_data[kriging_data[:, -1].argsort()]

        # set output by number of neighbours
        # TODO: account for a distance

        output_data = kriging_data[:self.max_no_of_observations]
        self.prepared_data = np.array(output_data)

        if verbose:
            print('Centroid of area id {} prepared for processing').format(
                unknown_area_id)

        return output_data

    # TODO: include weighted areal semivariance / covariance
    def normalize_weights(self, weights, estimated_value, kriging_type):
        """
        Algorithm for weight normalization to remove negative weights of the points which are
        clustering. Derived from Deutsch, C.V., Correcting for negative weights in
        ordinary kriging, Computers & Geosciences, Vol. 22, No. 7, pp. 765-773, 1996.

        :param: weights - weights matrix calculated with "normal" kriging procedure,
        :param: estimated_value - value estimated for a given, unknown point.
        :return: weight_matrix - normalized weight matrix where negative weights are removed and
        matrix is scalled to give a sum of all elements equal to 1.
        """

        if kriging_type == 'ord':
            weight_matrix = weights[:-1].copy()
            output_matrix = weights[:-1].copy()
        elif kriging_type == 'sim':
            weight_matrix = weights.copy()
            output_matrix = weights.copy()
        else:
            print('You did not choose any kriging type. Chosen type: <sim> - simple kriging.')
            weight_matrix = weights.copy()
            output_matrix = weights.copy()

        ###### Calculate average covariance between the location being ######
        ###### estimated and the locations with negative weights       ######

        locs = np.argwhere(weight_matrix < 0)  # Check where weights are below 0.0
        locs = locs[:, 0]

        # Calculate covariance between those points and unknown point
        if len(locs) >= 1:
            c = []
            mu = 0
            for i in locs:
                _c = estimated_value * self.prepared_data[i, 2]
                mu = mu + estimated_value + self.prepared_data[i, 2]
                c.append(_c)
                output_matrix[i, 0] = 0
            mu = mu / len(c)
            cov = np.sum(c) / len(c) - mu * mu

            ###### Calculate absolute magnitude of the negative weights #####

            w = weight_matrix[weight_matrix < 0]
            w = w.T
            magnitude = np.sum(np.abs(w)) / len(w)

            ###### Test values greater than 0 and check if they need to be
            ###### rescaled to 0 ######

            ###### if weight > 0 and Covariance between unknown point and known
            ###### point is less than the average covariance between the location
            ###### being estimated and the locations with negative weights and
            ###### and weight is less than absolute magnitude of the negative
            ###### weights then set weight to zero #####

            positive_locs = np.argwhere(weight_matrix > 0)  # Check where weights are greater than 0.0
            positive_locs = positive_locs[:, 0]

            for j in positive_locs:
                cov_est = (estimated_value * self.prepared_data[j, 2]) / 2
                mu = (estimated_value + self.prepared_data[j, 2]) / 2
                cov_est = cov_est - mu * mu
                if cov_est < cov:
                    if weight_matrix[j, 0] < magnitude:
                        output_matrix[j, 0] = 0

            ###### Normalize weight matrix to get a sum of all elements equal to 1 ######

            output_matrix = output_matrix / np.sum(output_matrix)

            return output_matrix
        else:
            return weights

    # Data processing private class methods

    def _join_datasets(self):
        """Function perform left join of two spatial datasets. Method is useful when someone is interested in
        the relation between point spatial data (population centroids) and polygons containing those points
        (counties).

        Both geodataframes must have the same coordinate reference system (crs). Test is performed before join.
        If crs is not the same then population centroids are transformed.

        OUTPUT:
        :return df: geodataframe of points with column indicating index from the counties geodataframe.
        """

        # Check crs
        if not pyproj.Proj(self.counties_dataset.crs).is_exact_same(pyproj.Proj(self.population_dataset.crs)):
            population = self.population_dataset.to_crs(self.counties_dataset.crs)
            df = sjoin(population, self.counties_dataset, how='left')
        else:
            df = sjoin(self.population_dataset, self.counties_dataset, how='left')
        return df

    def _get_tot_population(self, area_id_col, population_val_col):
        """Function calculate total population per administrative unit and returns dataframe with
        area index | sum of population

        INPUT:
        :param area_id_col: name of the id column from counties dataset,
        :param population_val_col: name of the column with population values in the population dataset.

        OUTPUT:
        :return tot_pop_series: series with total population per area.
        """
        tot_pop = self.joined_datasets.groupby([area_id_col]).sum()
        tot_pop_series = tot_pop[population_val_col]
        return tot_pop_series

    @staticmethod
    def _get_areal_centroids(counties, vals, ids):
        """Function get centroids and remove nan values from the dataset"""

        c = get_centroids(counties, vals, ids)
        c = c[~np.isnan(c).any(axis=1)]
        return c

    def _calculate_weighted_distances(self, unknown_area_id):
        """Function calculates weighted distances between unknown area and known areas"""

        dist_dict = self._prepare_distances_dict()
        base_area = dist_dict[unknown_area_id]
        base_area_list = base_area['coordinates']
        other_keys = list(dist_dict.keys())
        other_keys.remove(unknown_area_id)

        weighted_distances = []
        for k in other_keys:
            other_area_list = dist_dict[k]['coordinates']
            dist = calculate_block_to_block_distance(base_area_list,
                                                     other_area_list)
            weighted_distances.append([dist, k])
        return weighted_distances

    def _prepare_distances_dict(self, check_all=True, list_of_idx=None):
        """Function prepares dict with distances for weighted distance calculation
        between areas"""

        if list_of_idx is None:
            list_of_idx = []
        if check_all:
            # Copy all values
            new_d = self.joined_datasets.copy()
        else:
            # Copy selected values
            if len(list_of_idx) >= 1:
                new_d = self.joined_datasets[
                    self.joined_datasets[self.id_col].isin(
                        list_of_idx
                    )
                ].copy()
            else:
                raise ValueError(
                    "Minimum 1 id must be provided to prepare data"
                )

        new_d['px'] = new_d['geometry'].apply(lambda v: v[0].x)
        new_d['py'] = new_d['geometry'].apply(lambda v: v[0].y)

        new_dict = (new_d.groupby('IDx')
                    .apply(lambda v:
                           {'coordinates': list(map(list, zip(v['px'], v['py'], v['TOT'])))})
                    .to_dict())
        return new_dict

    @staticmethod
    def _get_list_from_dict(d, l):
        """Function creates list of lists from dict of dicts in the order
        given by the list with key names"""

        new_list = []

        for val in l:
            subdict = d[val]
            inner_list = []
            for subval in l:
                inner_list.append(subdict[subval])
            new_list.append(inner_list)

        return np.array(new_list)

    def _predict_value(self, predicted_array, k_array, vals_of_neigh_areas):

        w = np.linalg.solve(predicted_array, k_array)
        zhat = (np.matrix(vals_of_neigh_areas * w[:-1])[0, 0])

        if np.any(w < 0):

            # Normalize weights
            normalized_w = self.normalize_weights(w, zhat, 'ord')
            zhat = (np.matrix(vals_of_neigh_areas * normalized_w)[0, 0])

            sigmasq = (w.T * k_array)[0, 0]
            if sigmasq < 0:
                sigma = 0
            else:
                sigma = np.sqrt(sigmasq)
            return zhat, sigma, w[-1][0], normalized_w, self.unknown_area_id

        else:
            sigmasq = (w.T * k_array)[0, 0]
            if sigmasq < 0:
                sigma = 0
            else:
                sigma = np.sqrt(sigmasq)
            return zhat, sigma, w[-1][0], w, self.unknown_area_id

    # Modeling functions

    def poisson_kriging(self, pk_type='centroid'):
        """
        :param pk_type: available types:
            - 'ata' for area-to-area Poisson Kriging,
            - 'atp' for area-to-point Poisson Kriging,
            - 'centroid' for centroid based PK.

        To run kriging operation prepare_data method should be invoked first
        :return zhat, sigma, w[-1][0], w:
        [value in unknown location, error, estimated mean, weights, area_id]
        """

        vals_of_neigh_areas = self.prepared_data[:, 2]

        n = len(self.prepared_data)
        k = np.array([vals_of_neigh_areas])
        k = k.T
        k1 = np.matrix(1)
        k = np.concatenate((k, k1), axis=0)

        predicted=None

        if pk_type == 'centroid':

            # Calculation of centroid distances

            distances = calculate_distance(self.prepared_data[:, :2])
            predicted = self.model.predict(distances.ravel())

        elif pk_type == 'ata' or pk_type == 'atp':

            # Calculation of weighted distances

            distances = self._prepare_distances_dict(check_all=False,
                                                     list_of_idx=self.prepared_data[:, 3])
            calculated_distances = block_to_block_distances(distances)
            sorted_distances = self._get_list_from_dict(calculated_distances, self.prepared_data[:, 3])
            predicted = self.model.predict(sorted_distances.ravel())

        # Prepare predicted distances array

        predicted = np.matrix(predicted.reshape(n, n))

        # Add weights to predicted values (diagonal)

        weights_matrix = self.calculate_weight_arr()
        predicted = predicted + weights_matrix

        ones = np.matrix(np.ones(n))
        predicted = np.concatenate((predicted, ones.T), axis=1)
        ones = np.matrix(np.ones(n + 1))
        predicted = np.concatenate((predicted, ones), axis=0)

        prediction = self._predict_value(predicted, k, vals_of_neigh_areas)
        return prediction

    # Temporary functions

    def calculate_weight_arr(self):
        # Function used to create temp rates means - test purposes only

        vals_of_neigh_areas = self.prepared_data[:, 2]
        pop_of_neigh_areas = self.prepared_data[:, 4]

        weights_arr = np.mean(vals_of_neigh_areas) / pop_of_neigh_areas
        return np.diag(weights_arr)
