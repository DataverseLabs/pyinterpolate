import numpy as np
from helper_functions.euclidean_distance import calculate_distance


class Krige:

    def __init__(self, data, semivariogram_model):
        """
        :param data: dataset with known values and locations
        Each column should represent different dimension and the last column represents values
        example: [[dim_x1, dim_y1, val_1], [dim_x2, dim_y2, val_2]]
        :param semivariogram_model: semivariogram model returned by TheoreticalSemivariogram class
        """
        self.dataset = data
        self.model = semivariogram_model
        self.prepared_data = None

    def prepare_data(self, unknown_position, number_of_records=10):
        """
        :param unknown_position: array with position of unknown value (at this point only one location)
        :param number_of_records: number of the closest locations to the unknown position
        :return output_data: prepared dataset which contains:
        [[known_position_x, known_position_y, value, distance_to_unknown_position], [...]]
        """
        # Distances to unknown point
        r = np.array([unknown_position])
        known_positions = self.dataset[:, :-1]
        distances_array = np.zeros(known_positions.shape)
        for i in range(0, r.shape[1]):
            distances_array[:, i] = (known_positions[:, i] - r[:, i]) ** 2
        s = distances_array.sum(axis=1)
        s = np.sqrt(s)
        s = s.T

        # Build set for Kriging
        kriging_data = np.c_[self.dataset, s]
        kriging_data = kriging_data[kriging_data[:, -1].argsort()]
        output_data = kriging_data[:number_of_records]
        self.prepared_data = output_data
        return output_data

    def ordinary_kriging(self):
        """
        To run kriging operation prepare_data method should be invoked first
        :return zhat, sigma, w[-1][0], w:
        [value in unknown location, error, estimated mean, weights]
        """
        n = len(self.prepared_data)
        k = np.array([self.prepared_data[:, -1]])
        k = k.T
        k1 = np.matrix(1)
        k = np.concatenate((k, k1), axis=0)

        distances = calculate_distance(self.prepared_data[:, :-2])

        predicted = self.model.predict(distances.ravel())
        predicted = np.matrix(predicted.reshape(n, n))
        ones = np.matrix(np.ones(n))
        predicted = np.concatenate((predicted, ones.T), axis=1)
        ones = np.matrix(np.ones(n + 1))
        ones[0, n] = 0.0
        predicted = np.concatenate((predicted, ones), axis=0)

        w = np.linalg.solve(predicted, k)
        zhat = (np.matrix(self.prepared_data[:, -2] * w[:-1])[0, 0])
        sigmasq = (w.T * k)[0, 0]
        if sigmasq < 0:
            sigma = 0
        else:

            sigma = np.sqrt(sigmasq)
        return zhat, sigma, w[-1][0], w

    def simple_kriging(self, area_mean=None):
        """
        To run kriging operation prepare_data method should be invoked first
        :param area_mean: optional, if not given mean is calculated from points with known values
        :return zhat, sigma, area_mean, w
        [value in unknown location, error, mean calculated from the known values, weights]
        """
        n = len(self.prepared_data)

        if not area_mean:
            area_mean = np.sum(self.prepared_data[:, -2])
            area_mean = area_mean / n

        k = np.array([self.prepared_data[:, -1]])
        k = k.T

        distances = calculate_distance(self.prepared_data[:, :-2])

        K = self.model.predict(distances.ravel())
        K = np.matrix(K.reshape(n, n))

        w = np.linalg.solve(K, k)
        R = self.prepared_data[:, -2] - area_mean
        zhat = (np.matrix(R) * w)[0, 0]
        zhat += area_mean
        sigmasq = (w.T * k)[0, 0]

        if sigmasq < 0:
            sigma = 0
        else:
            sigma = np.sqrt(sigmasq)

        return zhat, sigma, area_mean, w

