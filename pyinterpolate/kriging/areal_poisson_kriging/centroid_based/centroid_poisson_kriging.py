import numpy as np

from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.transform.prepare_kriging_data import prepare_poisson_kriging_data


class CentroidPoissonKriging:

    def __init__(self, semivariogram_model, known_areas, known_areas_points):
        """
        Class performs centroid-based Poisson Kriging of areas.

        :param semivariogram_model: (TheoreticalSemivariance object) Semivariogram model fitted to the
            TheoreticalSemivariance class,
        :param known_areas: (numpy array) areas in the form:
            [area_id, polygon, centroid x, centroid y, value]
        :param known_areas_points: (numpy array) array of points within areas in the form:
            [area_id, [point_position_x, point_position_y, value]],
        """

        self.model = semivariogram_model
        self.known_areas = known_areas
        self.known_areas = self.known_areas[self.known_areas[:, 0].argsort()]  # Sort by id
        self.known_areas_points = known_areas_points
        self.known_areas_points = self.known_areas_points[self.known_areas_points[:, 0].argsort()]  # Sort by id

        self.prepared_data = None

    def predict(self, unknown_area, unknown_area_points,
                number_of_neighbours, max_search_radius, weighted,
                test_anomalies=True):
        """
        Function predicts areal value in a unknown location based on the centroid-based Poisson Kriging.

        INPUT:

        :param unknown_area: (numpy array) unknown area data in the form:
            [area_id, polygon, centroid x, centroid y]
        :param unknown_area_points: (numpy array) points within an unknown area in the form:
            [area_id, [point_position_x, point_position_y, value]]
        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to
            the number_of_neighbours).
        :param weighted: (bool) distances weighted by population (True) or not (False),
        :param test_anomalies: (bool) check if weights are negative.

        OUTPUT:

        :return: prediction, error, estimated mean, weights:
            [value_in_unknown_area, error, estimated_mean, weights]
        """

        self.prepared_data = prepare_poisson_kriging_data(
            unknown_area=unknown_area, points_within_unknown_area=unknown_area_points,
            known_areas=self.known_areas, points_within_known_areas=self.known_areas_points,
            number_of_neighbours=number_of_neighbours, max_search_radius=max_search_radius,
            weighted=weighted
        )  # [id (known), coo_x, coo_y, val, dist_to_unknown, total population]

        n = self.prepared_data.shape[0]
        unknown_distances = self.prepared_data[:, -2]
        k = self.model.predict(unknown_distances)  # predicted values from distances to unknown
        k = k.T
        k_ones = np.ones(1)[0]
        k = np.r_[k, k_ones]

        data_for_distance = self.prepared_data[:, 1:3]
        dists = calc_point_to_point_distance(data_for_distance)

        predicted_weights = self.model.predict(dists.ravel())
        predicted = np.array(predicted_weights.reshape(n, n))

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

        zhat = self.prepared_data[:, 3].dot(w[:-1])

        # Test for anomalies
        if test_anomalies:
            if zhat < 0:
                user_input_message = 'Estimated value is below zero and it is: {}. \n'.format(zhat)
                text_error = user_input_message + 'Program is terminated. Try different semivariogram model. ' \
                                                  '(Did you use gaussian model? \
                            If so then try to use other models like linear or exponential) and/or analyze your data \
                            for any clusters which may affect the final estimation'

                raise ValueError(text_error)

        sigmasq = (w.T * k)[0]
        if sigmasq < 0:
            sigma = 0
        else:
            sigma = np.sqrt(sigmasq)
        return zhat, sigma, w[-1], w

    def calculate_weight_arr(self):
        """
        Function calculates additional weights for the diagonal matrix of predicted values.
        delta * (m' / n(u_i))
        where:
        delta is a param which indicates if this is observed area,
        m'/n(u_i) - population weighted mean of N rates
        """

        vals_of_neigh_areas = self.prepared_data[:, 3]
        pop_of_neigh_areas = self.prepared_data[:, -1]

        weighted = np.sum(vals_of_neigh_areas * pop_of_neigh_areas)
        weights_arr = weighted / np.sum(pop_of_neigh_areas)
        w = np.ones(shape=vals_of_neigh_areas.shape)
        w = (weights_arr * w)
        return np.diag(w)
