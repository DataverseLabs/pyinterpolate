import numpy as np

from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance
from pyinterpolate.kriging.helper_functions.euclidean_distance import block_to_block_distances
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_block_to_block_distance


# TODO: remove matrix data structures

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

    def __init__(self):
        """
        Class for calculation of poisson kriging

        """
        self.model = None
        self.joined_datasets = None
        self.id_col = None
        self.val_col = None
        self.pop_col = None
        self.total_population_per_unit = None
        self.centroids_of_areal_data = None
        self.prepared_data = None
        self.unknown_area_id = None

        # Parameters
        self.lags = None
        self.step = None
        self.min_no_of_observations = None
        self.max_search_radius = None

    # Data preparation functions

    def set_params(self, model,
                   joined_datasets, population_series, centroids_dataset,
                   id_col, val_col, pop_col,
                   lags_number, lag_step_size,
                   min_no_of_observations, search_radius):

        self.model = model
        self.joined_datasets = joined_datasets
        self.total_population_per_unit = population_series
        self.centroids_of_areal_data = centroids_dataset
        self.id_col = id_col
        self.val_col = val_col
        self.pop_col = pop_col
        self.lags = lags_number
        self.step = lag_step_size
        self.min_no_of_observations = min_no_of_observations
        self.max_search_radius = search_radius
        print('Parameters have been set')

    def prepare_prediction_data(self, unknown_areal_data_row, unknown_areal_data_centroid,
                                weighted=False, verbose=False):
        """
        Function prepares data from unknown locations for Poisson Kriging.
        :param unknown_areal_data: PKData object (row) with areal and population data.
        :param weighted: distances weighted by population (True) or not (False),
        :param verbose: if True then method informs about the successful operation.
        :return prediction: prepared dataset which contains:
        [[x, y, value, known_area_id, distance_to_unknown_position], [...]],
        """

        areal_id = unknown_areal_data_centroid[0][-1]

        cx_cy = unknown_areal_data_centroid[0][:2]
        r = np.array([cx_cy])
        known_centroids = self.centroids_of_areal_data
        kc = known_centroids[:, :2]

        # Build set for Poisson Kriging

        if weighted:
            weighted_distances = self._calculate_weighted_distances(unknown_areal_data_row,
                                                                    areal_id)
            s = []
            for wd in weighted_distances:
                for k in known_centroids:
                    if wd[1] in k:
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

        # set output by distance params

        # search radius

        max_search_pos = np.argmax(kriging_data[:, -1] > self.max_search_radius)
        output_data = kriging_data[:max_search_pos]

        # check number of observations

        if len(output_data) < self.min_no_of_observations:
            output_data = kriging_data[:self.min_no_of_observations]
            # TODO: info to the app logs
            # print('Dataset has been set based on the minimum number of observations')

        # set final dataset

        self.prepared_data = output_data
        if verbose:
            print('Predictions data prepared')

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

    def _calculate_weighted_distances(self, unknown_area, unknown_area_id):
        """Function calculates weighted distances between unknown area and known areas"""

        dist_dict = self._prepare_distances_dict(unknown_area)
        base_area = dist_dict[unknown_area_id]
        base_area_list = base_area['coordinates']
        other_keys = list(dist_dict.keys())

        weighted_distances = []
        for k in other_keys:
            other_area_list = dist_dict[k]['coordinates']
            dist = calculate_block_to_block_distance(base_area_list,
                                                     other_area_list)
            weighted_distances.append([dist, k])
        return weighted_distances

    def _prepare_distances_dict(self, unknown_area):
        """Function prepares dict with distances for weighted distance calculation
        between areas"""

        new_d = self.joined_datasets.copy()
        new_d = new_d.append(unknown_area, ignore_index=True)

        try:
            new_d['px'] = new_d['geometry'].apply(lambda v: v[0].x)
            new_d['py'] = new_d['geometry'].apply(lambda v: v[0].y)
        except TypeError:
            new_d['px'] = new_d['geometry'].apply(lambda v: v.x)
            new_d['py'] = new_d['geometry'].apply(lambda v: v.y)

        new_dict = (new_d.groupby(self.id_col)
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
                print(sigmasq)
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

        predicted = None

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

    # Population-based weights array (for m' parameter)

    def calculate_weight_arr(self):

        vals_of_neigh_areas = self.prepared_data[:, 2]
        pop_of_neigh_areas = self.prepared_data[:, 4]

        weighted = np.sum(vals_of_neigh_areas * pop_of_neigh_areas)
        weights_arr = weighted / np.sum(pop_of_neigh_areas)
        w = np.ones(shape=vals_of_neigh_areas.shape)
        w = (weights_arr * w) / pop_of_neigh_areas
        return np.diag(w)
