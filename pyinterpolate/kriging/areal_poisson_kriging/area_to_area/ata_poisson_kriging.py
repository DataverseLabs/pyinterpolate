import numpy as np

from pyinterpolate.transform.prepare_kriging_data import prepare_ata_known_areas
from pyinterpolate.transform.prepare_kriging_data import prepare_ata_data
from pyinterpolate.transform.prepare_kriging_data import prepare_distances_list_unknown_area


class WeightedBlock2BlockSemivariance:

    def __init__(self, semivariance_model):
        self.semivariance_model = semivariance_model

    def _avg_smv(self, datarows):
        """
        Function calculates weight and partial semivariance of a data row [point value a, point value b, distance]
        :param datarows: (numpy arrays) numpy arrays of [point value a, point value b, distance between points]
        :return: (float, float) weighted semivariance and single weight
        """
        weighted_semivariance_sum = 0
        sum_of_weights = 0

        for datarow in datarows:
            single_weight = datarow[0] * datarow[1]
            partial_semivar = self.semivariance_model.predict(datarow[-1])
            weighted_semivar = partial_semivar * single_weight
            weighted_semivariance_sum = weighted_semivariance_sum + weighted_semivar
            sum_of_weights = sum_of_weights + single_weight

        return weighted_semivariance_sum, sum_of_weights

    def calculate_average_semivariance(self, data_points):
        """
        Function calculates average semivariance. Calculation is performed based on the equation:

            (1 / SUM(Pi-s=1) SUM(Pj-s'=1) w_(ss')) * SUM(Pi-s=1) SUM(Pj-s'=1) w_(ss') * semivariance(distance)

        :return: (numpy array) array of predicted and weighted semivariances
        """
        k = []
        for prediction_input in data_points:
            w_sem_sum = 0
            w_sem_weights_sum = 0
            for single_area in prediction_input:
                w_sem = self._avg_smv(single_area)
                w_sem_sum = w_sem_sum + w_sem[0]
                w_sem_weights_sum = w_sem_weights_sum + w_sem[1]
            k.append(w_sem_sum / w_sem_weights_sum)
        return np.array(k)


class AtAPoissonKriging:

    def __init__(self, regularized_model=None, known_areas=None, known_areas_points=None):
        """

        :param regularized_model: (RegularizedSemivariogram object) Semivariogram model after deconvolution procedure,
        :param known_areas: (numpy array) array of areas in the form:
            [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value]
        :param known_areas_points: (numpy array) array of points within areas in the form:
            [area_id, [point_position_x, point_position_y, value]]
        """

        self.model = regularized_model
        self.known_areas = known_areas
        self.known_areas = self.known_areas[self.known_areas[:, 0].argsort()]  # Sort by id
        self.known_areas_points = known_areas_points
        self.known_areas_points = self.known_areas_points[self.known_areas_points[:, 0].argsort()]  # Sort by id

        self.block_to_block_smv = None
        self.prepared_data = None

    def predict(self, unknown_location_points,
                number_of_neighbours, max_search_radius):
        """
        Function predicts areal value in a unknown location based on the centroid-based Poisson Kriging
        :param unknown_location_points: (numpy array) array of points within an unknown area in the form:
            [area_id, [point_position_x, point_position_y, value]]
        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to number of
            neighbors).
        :return: prediction, error, estimated mean, weights:
            [value in unknown location, error, estimated mean, weights]
        """

        self.prepared_data = prepare_ata_data(
            points_within_unknown_area=unknown_location_points,
            known_areas=self.known_areas, points_within_known_areas=self.known_areas_points,
            number_of_neighbours=number_of_neighbours, max_search_radius=max_search_radius
        )  # [id (known), val, [known pt val, unknown pt val, distance between points], total points value]

        self.block_to_block_smv = WeightedBlock2BlockSemivariance(semivariance_model=self.model)

        # Calculate Average Point to Point semivariance for each area - len of dataset
        areas_and_points = self.prepared_data[:, 2]
        k = self.block_to_block_smv.calculate_average_semivariance(areas_and_points)
        k = k.T
        k_ones = np.ones(1)[0]
        k = np.r_[k, k_ones]

        # Calculate block to block average semivariance between known areas

        # Prepare areas blocks for calculation

        block_ids = self.prepared_data[:, 0]

        points_of_interest = []
        for area_id in block_ids:
            points_of_interest.append(self.known_areas_points[self.known_areas_points[:, 0] == area_id])

        # Calculate distances between blocks / prepare data for prediction
        # [id base,
        #     [
        #         id other,
        #         [base point value, other point value, distance between points]
        #     ]
        #  ]

        distances_of_chosen_areas = prepare_ata_known_areas(points_of_interest)

        predicted = self._calculate_avg_semivars_between_known(distances_of_chosen_areas)

        # Add weights to predicted values (diagonal)

        weights_mtx = self.calculate_weight_arr()
        predicted = predicted + weights_mtx

        # Prepare weights matrix

        p_ones = np.ones((predicted.shape[0], 1))
        predicted_with_ones_col = np.c_[predicted, p_ones]
        p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
        p_ones_row[0][-1] = 0.
        weights = np.r_[predicted_with_ones_col, p_ones_row]

        # Solve Kriging system
        try:
            w = np.linalg.solve(weights, k)
        except TypeError:
            weights = weights.astype(np.float)
            k = k.astype(np.float)
            w = np.linalg.solve(weights, k)

        zhat = self.prepared_data[:, 1].dot(w[:-1])

        # Calculate error TODO Check if it distance are more stable or not with smv_inblock_unknown
        dists_unknown_inblock = prepare_distances_list_unknown_area(unknown_location_points[1])
        smv_inblock_unknown = self.block_to_block_smv.calculate_average_semivariance([dists_unknown_inblock])

        sig_base = (w.T * k)[0]
        sigmasq = smv_inblock_unknown[0] - sig_base
        if sigmasq < 0:
            sigma = 0
        else:
            sigma = np.sqrt(sigmasq)
        return zhat, sigma, w[-1], w

    def calculate_weight_arr(self):
        """Function calculates additional weights for the diagonal matrix of predicted values"""

        vals_of_neigh_areas = self.prepared_data[:, 1]
        pop_of_neigh_areas = self.prepared_data[:, -1]

        weighted = np.sum(vals_of_neigh_areas * pop_of_neigh_areas)
        weights_arr = weighted / np.sum(pop_of_neigh_areas)
        w = np.ones(shape=vals_of_neigh_areas.shape)
        w = (weights_arr * w) / pop_of_neigh_areas
        return np.diag(w)

    def _calculate_avg_semivars_between_known(self, list_of_distances):
        """
        Function calculates average semivariance between known areas
        :param list_of_distances: (numpy array)
            [id base,
                [
                    id other,
                    [base point value, other point value, distance between points]
                ]
            ]
        :return: (numpy array) array of predicted and weighted semivariances
        """

        semivariances = []
        for area in list_of_distances:
            points_group = area[1]
            points_group = [x[1] for x in points_group]
            smv = self.block_to_block_smv.calculate_average_semivariance(points_group)
            semivariances.append(smv)
        return np.array(semivariances)
