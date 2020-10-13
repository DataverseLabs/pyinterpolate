import numpy as np

from pyinterpolate.data_processing.data_transformation.prepare_kriging_data import prepare_kriging_data
from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance


class Krige:
    """
    Class for kriging interpolation of the unknown values in a given location (position). Class takes two arguments
    during the initialization:
    semivariogram_model - semivariogram model,
    known_points - array of known values [x, y, val]

    Available methods:

    - ordinary_kriging - ordinary kriging of unknown point value,
    - simple_kriging - simple kriging of unknown point value.

    Method may raise value error if estimated value is below 0. You may use try: statement to overwrite those values
        with some constant or NaN or you could use different semivariogram model. Sometimes this problem is related to
        the input data, especially clusters of points. In this case aggregate those clusters and then estimate
        semivariogram and perform kriging.

    INITLIALIZATION PARAMS:

    :param semivariogram_model: semivariogram model returned by TheoreticalSemivariogram class
    :param known_points: dataset with known values and locations. Each column should represent different dimension and
        the last column represents values example: [[dim_x1, dim_y1, val_1], [dim_x2, dim_y2, val_2]].
    """

    def __init__(self, semivariogram_model, known_points):
        """
        INPUT:

        :param semivariogram_model: semivariogram model returned by TheoreticalSemivariogram class
        :param known_points: dataset with known values and locations

        Each column should represent different dimension and the last column represents values
        example: [[dim_x1, dim_y1, val_1], [dim_x2, dim_y2, val_2]]
        """

        self.dataset = known_points
        self.model = semivariogram_model
        self.distances = None

    def ordinary_kriging(self, unknown_location, number_of_neighbours, test_anomalies=True):
        """
        Function predicts value at unknown location.

        INPUT:

        :param unknown_location: (tuple) position of unknown location,
        :param number_of_neighbours: (int) number of the closest locations to the unknown position which should be
            included in the modeling,
        :param test_anomalies: (bool) check if weights are negative.

        OUTPUT:

        :return:
            for ordinary kriging:

                - zhat, sigma, w[-1][0], w == [value in unknown location, error, estimated mean, weights]

            for simple kriging:

                - zhat, sigma, area_mean, w == [value in unknown location, error, mean, weights]
        """

        prepared_data = prepare_kriging_data(unknown_position=unknown_location,
                                             data_array=self.dataset,
                                             number_of_neighbours=number_of_neighbours)
        n = number_of_neighbours
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

    def simple_kriging(self, unknown_location, number_of_neighbours, mu=None, test_anomalies=True):
        """
        Function predicts value at unknown location.

        INPUT:

        :param unknown_location: (tuple) position of unknown location,
        :param number_of_neighbours: (int) number of the closest locations to the unknown position which should be
            included in the modeling,
        :param mu: (float) global mean which should be known before processing. If not given then it is calculated
            from the sample but then it may cause a relative large errors (this mean is expectation of the random field,
            so without knowledge of the ongoing processes it is unknown).
        :param test_anomalies: (bool) check if weights are negative.

        OUTPUT:

        :return:
            for ordinary kriging:

                - zhat, sigma, w[-1][0], w == [value in unknown location, error, estimated mean, weights]

            for simple kriging:

                - zhat, sigma, area_mean, w == [value in unknown location, error, mean, weights]
        """

        prepared_data = prepare_kriging_data(unknown_position=unknown_location,
                                             data_array=self.dataset,
                                             number_of_neighbours=number_of_neighbours)
        n = number_of_neighbours

        if mu is None:
            vals = self.dataset[:, -1]
            mu = np.sum(vals)
            mu = mu / len(vals)

        unknown_distances = prepared_data[:, -1]
        k = self.model.predict(unknown_distances)
        k = k.T

        dists = calc_point_to_point_distance(prepared_data[:, :-2])
        predicted_weights = self.model.predict(dists.ravel())
        predicted = np.array(predicted_weights.reshape(n, n))

        w = np.linalg.solve(predicted, k)
        r = prepared_data[:, -2] - mu
        zhat = r.dot(w)
        zhat = zhat + mu

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
        return zhat, sigma, mu, w
