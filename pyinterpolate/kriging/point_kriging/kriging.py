import numpy as np

from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.transform.prepare_kriging_data import prepare_kriging_data
from pyinterpolate.transform.tests import does_variogram_exist


class Krige:
    """
    Class for kriging interpolation of the unknown values at a given location (position). Class takes two arguments
    during the initialization:
    semivariogram_model - semivariogram model,
    known_points - array of known values [x, y, val]

    Available methods:

    - ordinary_kriging - ordinary kriging of unknown point value,
    - simple_kriging - simple kriging of unknown point value.

    Class methods may raise ValueError if estimated value is below 0. You may use try: ... except: ... statement to
        overwrite those values with some constant or NaN or you could use different semivariogram model. Sometimes this
        problem is related to the input data, especially to the clustered groups of points. In this case aggregate
        those clusters and then estimate semivariogram and perform kriging.

    INITLIALIZATION PARAMS:

    :param semivariogram_model: semivariogram model returned by TheoreticalSemivariogram class
    :param known_points: dataset with known values and locations. Each column should represent different dimension and
        the last column represents values example: [[dim_x1, dim_y1, val_1], [dim_x2, dim_y2, val_2]].
    """

    def __init__(self, semivariogram_model, known_points):
        """
        INPUT:

        :param semivariogram_model: (TheoreticalSemivariogram) Theoretical Semivariogram used for data interpolation,
        :param known_points: (numpy array) dataset with known values and locations. Each column should represent
            different dimension and the last column represents values:
            [[dim_x1, dim_y1, val_1], [dim_x2, dim_y2, val_2]]
        """

        self.dataset = known_points

        # Test semivariogram model
        does_variogram_exist(semivariogram_model)
        self.model = semivariogram_model
        self.distances = None

    def ordinary_kriging(self,
                         unknown_location,
                         neighbors_range=None,
                         min_no_neighbors=1,
                         max_no_neighbors=256,
                         test_anomalies=True):
        """
        Function predicts value at unknown location with Ordinary Kriging technique.

        INPUT:

        :param unknown_location: (tuple) position of unknown location,
        :param neighbors_range: (float) distance for each neighbors are affecting the interpolated point, if not given
            then it is set to the semivariogram range,
        :param min_no_neighbors: (int) minimum number of neighbors used for interpolation if there is not any neighbor
            within the range specified by neighbors_range,
        :param max_no_neighbors: (int) maximum number of n-closest neighbors used for interpolation if there are too
            many neighbors in range. It speeds up calculations for large datasets.
        :param test_anomalies: (bool) check if weights are negative.

        OUTPUT:

        :return: predicted, error, estimated mean, weights
            [value_in_unknown_location, error, estimated_mean, weights]
        """

        # Check range
        if neighbors_range is None:
            neighbors_range = self.model.range

        prepared_data = prepare_kriging_data(unknown_position=unknown_location,
                                             data_array=self.dataset,
                                             neighbors_range=neighbors_range,
                                             min_number_of_neighbors=min_no_neighbors,
                                             max_number_of_neighbors=max_no_neighbors)
        n = len(prepared_data)
        unknown_distances = prepared_data[:, -1]
        k = self.model.predict(unknown_distances)
        k = k.T
        k_ones = np.ones(1)[0]
        k = np.r_[k, k_ones]

        dists = calc_point_to_point_distance(prepared_data[:, :-2])

        predicted_weights = self.model.predict(dists.ravel())
        predicted = np.array(predicted_weights.reshape(n, n))
        p_ones = np.ones((predicted.shape[0], 1))
        predicted_with_ones_col = np.c_[predicted, p_ones]
        p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
        p_ones_row[0][-1] = 0.
        weights = np.r_[predicted_with_ones_col, p_ones_row]

        w = np.linalg.solve(weights, k)
        zhat = prepared_data[:, -2].dot(w[:-1])

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

    def simple_kriging(self,
                       unknown_location,
                       global_mean,
                       neighbors_range=None,
                       min_no_neighbors=1,
                       max_no_neighbors=256,
                       test_anomalies=True):
        """
        Function predicts value at unknown location with Simple Kriging technique.

        INPUT:

        :param unknown_location: (tuple) position of unknown location,
        :param neighbors_range: (float) distance for each neighbors are affecting the interpolated point, if not given
            then it is set to the semivariogram range,
        :param min_no_neighbors: (int) minimum number of neighbors used for interpolation if there is not any neighbor
            within the range specified by neighbors_range,
        :param global_mean: (float) global mean which should be known before processing,
        :param max_no_neighbors: (int) maximum number of n-closest neighbors used for interpolation if there are too
            many neighbors in range. It speeds up calculations for large datasets.
        :param test_anomalies: (bool) check if weights are negative.

        OUTPUT:

        :return: predicted, error, mean, weights:
            [value_in_unknown_location, error, mean, weights]
        """

        # Check range
        if neighbors_range is None:
            neighbors_range = self.model.range

        prepared_data = prepare_kriging_data(unknown_position=unknown_location,
                                             data_array=self.dataset,
                                             neighbors_range=neighbors_range,
                                             min_number_of_neighbors=min_no_neighbors,
                                             max_number_of_neighbors=max_no_neighbors)
        n = len(prepared_data)

        unknown_distances = prepared_data[:, -1]
        k = self.model.predict(unknown_distances)
        k = k.T

        dists = calc_point_to_point_distance(prepared_data[:, :-2])
        predicted_weights = self.model.predict(dists.ravel())
        predicted = np.array(predicted_weights.reshape(n, n))

        w = np.linalg.solve(predicted, k)
        r = prepared_data[:, -2] - global_mean
        zhat = r.dot(w)
        zhat = zhat + global_mean

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
        return zhat, sigma, global_mean, w
