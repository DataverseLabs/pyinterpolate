# Base libraries
import numpy as np

# Data vizualization libraries
import matplotlib.pyplot as plt

# Pyinterpolate libraries
from pyinterpolate.semivariance.areal_semivariance.areal_semivariance import ArealSemivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance  # Experimental semivariogram


class RegularizedSemivariogram:
    """
    Class performs deconvolution of semivariogram of areal data.
    """

    def __init__(self):

        # Procedure control parameters
        self.iter = 0
        self.max_iters = None

        self.deviation_gain = 100
        self.min_deviation_gain = None

        self.diff_d_stat = 100
        self.min_diff_d_stat = None

        self.const_d_stat_reps = 0
        self.min_diff_d_stat_reps = None

        self.weight_error_lags = False

        self.weight_change = False

        # Regularization parameters
        self.ranges = None
        self.weights = []
        self.deviations = []

        # Initial semivariogram models and parameters
        self.experimental_semivariogram_of_areal_data = None
        self.initial_theoretical_model_of_areal_data = None
        self.initial_regularized_model = None
        self.initial_deviation = None

        # Temporary semivariogram models and class parameters
        self.temp_empirical_semivariogram = None
        self.temp_theoretical_semivariogram_model = None
        self.temp_regularized_model = None
        self.temp_deviation = None

        # Optimal semivariogram models and params
        self.optimal_empirical_semivariogram = None
        self.optimal_theoretical_model = None
        self.optimal_regularized_model = None
        self.optimal_deviation = None

        # Data
        self.areal_data = None
        self.areal_lags = None
        self.areal_step_size = None
        self.point_support_data = None

    def _regularize(self, empirical_semivariance, semivariance_model):

        # Initialize areal semivariance object
        areal_semivariance = ArealSemivariance(self.areal_data,
                                               self.areal_lags,
                                               self.areal_step_size,
                                               self.point_support_data,
                                               weighted_semivariance=self.weight_error_lags)

        # Regularize semivariogram of areal data
        theoretically_regularized_model = areal_semivariance.regularize_semivariogram(
            empirical_semivariance=empirical_semivariance,
            theoretical_semivariance_model=semivariance_model)

        return theoretically_regularized_model

    @staticmethod
    def _calculate_deviation(regularized_model, theoretical_model):
        """
        Function calculates deviation between experimental and theoretical semivariogram over given lags.

        INPUT:
        :param regularized_model: (numpy array) array of the values generated for the regularized model,
        :param theoretical_model: (TheoreticalSemivariance) theoretical model of data,

        OUTPUT:
        :return deviation: (float) scalar which describes deviation between semivariograms.
        """

        theoretical_values = theoretical_model.predict(regularized_model[:, 0])
        regularized_values = regularized_model[:, 1]

        deviation = np.abs(regularized_values - theoretical_values)
        deviation = np.divide(deviation,
                              theoretical_values,
                              out=np.zeros_like(deviation),
                              where=theoretical_values != 0)
        deviation = np.mean(deviation)
        return deviation

    def _rescale_optimal_point_support(self):
        """Function rescales the optimal point support model and creates new experimental values for
        each lag based on the equation:
            y(1)(h) = y_opt(h) x w(h)
            w(h) = 1 + [(y_exp_v(h) - y_opt_v(h) / (s^2 * sqrt(iter))]
            s = sill of the model y_exp_v
            iter = iteration number

        OUTPUT:
        :return rescalled_point_support_semivariogram: (numpy array) of the form [[lag, semivariance],
                                                                                  [lag_x, semivariance_x],
                                                                                  [..., ...]]
        """

        sill = self.initial_theoretical_model_of_areal_data.params[1]
        lags = self.optimal_regularized_model[:, 0]

        y_opt_h = self.optimal_theoretical_model.predict(lags)

        if not self.weight_change:
            denominator = sill ** 2 * np.sqrt(self.iter)

            y_exp_v_h = self.initial_theoretical_model_of_areal_data.predict(lags)

            y_opt_v_h = self.optimal_regularized_model[:, 1]

            numerator = (y_exp_v_h - y_opt_v_h)

            w = 1 + numerator / denominator
        else:
            w = 1 + ((self.weights[-1] - 1) / 2)

        rescalled = self.experimental_semivariogram_of_areal_data.copy()
        rescalled[:, 1] = y_opt_h * w

        return rescalled, w

    def _check_deviation_gain(self):
        if self.deviation_gain <= self.min_deviation_gain:
            return True
        else:
            return False

    def _check_loop_limit(self):
        if self.iter >= self.max_iters:
            return True
        else:
            return False

    def _check_diff_d_stat(self):
        if self.diff_d_stat < self.min_diff_d_stat:
            if self.const_d_stat_reps >= self.min_diff_d_stat_reps:
                return True
            else:
                self.const_d_stat_reps += 1
                return False
        else:
            return False

    def _check_algorithm(self):
        t1 = self._check_deviation_gain()  # Default False
        t2 = self._check_diff_d_stat()  # Default False
        t3 = self._check_loop_limit()  # Default False

        cond = not (t1 or t2 or t3)  # Default False

        return cond

    def fit(self, areal_data, areal_lags, areal_step_size,
            point_support_data, ranges=None, weighted_lags=True):

        # Update data class params
        self.areal_data = areal_data
        self.areal_lags = areal_lags
        self.areal_step_size = areal_step_size
        self.point_support_data = point_support_data

        self.weight_error_lags = weighted_lags

        if ranges is None:
            self.ranges = len(areal_lags)
        else:
            self.ranges = ranges

        # Compute experimental semivariogram of areal data from areal centroids

        areal_centroids = areal_data[:, 2:]

        self.experimental_semivariogram_of_areal_data = calculate_semivariance(
            areal_centroids,
            areal_lags,
            areal_step_size
        )

        # Compute theoretical semivariogram of areal data from areal centroids

        self.initial_theoretical_model_of_areal_data = TheoreticalSemivariogram(
            areal_centroids,
            self.experimental_semivariogram_of_areal_data
        )

        self.initial_theoretical_model_of_areal_data.find_optimal_model(
            weighted=weighted_lags,
            number_of_ranges=self.ranges
        )

        # Regularize model
        self.initial_regularized_model = self._regularize(self.experimental_semivariogram_of_areal_data,
                                                          self.initial_theoretical_model_of_areal_data)

        # Calculate d-stat
        self.initial_deviation = self._calculate_deviation(self.initial_regularized_model,
                                                           self.initial_theoretical_model_of_areal_data)

        self.deviations.append(self.initial_deviation)

    def transform(self, max_iters=25, min_deviation_gain=0.05, min_diff_d_stat=0.01, min_diff_d_stat_reps=3):

        # Check if data is fitted
        if self.initial_regularized_model is None:
            raise RuntimeError('Before transform you must fit areal data and calculate initial point support models')

        # Update class control params
        self.iter = 1
        self.max_iters = max_iters

        self.min_deviation_gain = min_deviation_gain

        self.min_diff_d_stat = min_diff_d_stat
        self.min_diff_d_stat_reps = min_diff_d_stat_reps

        # Update initial optimal models
        self.optimal_theoretical_model = self.initial_theoretical_model_of_areal_data
        self.optimal_regularized_model = self.initial_regularized_model
        self.optimal_deviation = self.initial_deviation

        # Prepare semivariogram modeling data
        areal_centroids = self.areal_data[:, 2:]
        ranges = self.ranges
        is_weighted = self.weight_error_lags

        # Start iteration procedure

        while self._check_algorithm():
            # Compute new experimental values for new experimental point support model

            self.temp_empirical_semivariogram, weights = self._rescale_optimal_point_support()
            self.weights.append(weights)

            # Fit rescaled empirical semivariogram to the new theoretical function
            self.temp_theoretical_semivariogram_model = TheoreticalSemivariogram(areal_centroids,
                                                                                 self.temp_empirical_semivariogram)
            self.temp_theoretical_semivariogram_model.find_optimal_model(
                weighted=is_weighted,
                number_of_ranges=ranges
            )

            # Regularize model
            self.temp_regularized_model = self._regularize(
                self.temp_empirical_semivariogram,
                self.temp_theoretical_semivariogram_model
            )

            # Compute difference statistics

            self.temp_deviation = self._calculate_deviation(self.temp_regularized_model,
                                                            self.temp_theoretical_semivariogram_model)

            # Analyze deviations
            self.optimal_regularized_model = self.temp_regularized_model

            if self.temp_deviation < self.optimal_deviation:
                self.weight_change = False

                self.optimal_empirical_semivariogram = self.temp_theoretical_semivariogram_model

                self.deviation_gain = np.abs(self.temp_deviation - self.optimal_deviation) / self.optimal_deviation
                self.diff_d_stat = self.temp_deviation / self.optimal_deviation

                self.optimal_deviation = self.temp_deviation

            else:
                self.weight_change = True

            self.deviations.append(self.optimal_deviation)
            self.iter = self.iter + 1

        return self.optimal_regularized_model, self.optimal_deviation

    def show_semivariograms(self):
        """
        Function shows experimental semivariogram, theoretical semivariogram and regularized semivariogram after
        semivariogram regularization.
        """
        lags = self.experimental_semivariogram_of_areal_data[:, 0]
        plt.figure(figsize=(12, 12))
        plt.plot(lags,
                 self.experimental_semivariogram_of_areal_data[:, 1], color='b')
        plt.plot(lags,
                 self.optimal_theoretical_model.predict(lags), color='r',
                 linestyle='--')
        plt.plot(lags, self.optimal_regularized_model[:, 1], color='g',
                 linestyle='-.')
        plt.legend(['Empirical semivariogram', 'Theoretical semivariogram',
                    'Regularized semivariogram, iteration {}'.format(self.iter)])
        plt.title('Semivariograms comparison. Deviation value: {}'.format(self.optimal_deviation))
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()
