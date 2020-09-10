# Base libraries
import numpy as np

# Data vizualization libraries
import matplotlib.pyplot as plt

# Pyinterpolate libraries
from pyinterpolate.semivariance.areal_semivariance.areal_semivariance import ArealSemivariance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram

# Verbose
from .verbose_sem_dec import SemivariogramDeconvolutionMessages


class RegularizedSemivariogram:
    """
    Class performs deconvolution of semivariogram of areal data.

        METHODS:
        regularize_model: method regularizes given areal model based on the:
            a) data with areal counts of some variable,
            b) data with population units and counts (divided per area),
            Based on the experimental semivariogram of areal centroids and population units function performs
            deconvolution and returns theoretical model for given areas.
            Method is described in: Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular
            Geographical Units, Mathematical Geology 40(1), 101-128, 2008.,
        rescale: function rescales the optimal point support model and creates new experimental values for
            each lag,
        calculate_deviation: function calculates deviation between experimental and theoretical semivariogram
            over given lags,
        show_semivariograms: function shows experimental semivariogram, theoretical semivariogram and regularized
            semivariogram,
        show_deviation: plot of changes of deviation over time,
        _check_loops_status: method checks model loop's statistics to stop iteration after certain events occurs,
        _check_optimizer: function checks if analysis is not in local optimum.

    """

    def __init__(self,
                 d_statistics_change=0.02,
                 ranges=32,
                 loop_limit=20,
                 min_no_loops=5,
                 number_of_loops_with_const_mean=3,
                 mean_diff=0.005,
                 weighted_semivariance=False,
                 verbose=True):
        """
        All parameters control behavior of the algorithm.
        :param d_statistics_change: (float) min change of deviation, if change is smaller then algorithm is stopped,
        :param ranges:(int) numberof ranges to search for optimal model,
        :param loop_limit: (int) max number of algorithm iterations,
        :param min_no_loops: (int) min number of loops,
        :param number_of_loops_with_const_mean: (int) number of times when devaition is not changing or change is very
            small (mean_diff parameter).
        :param mean_diff: (float) difference between deviations between the loops, if smaller than parameter
            then algorithm is stopped. This situation must occur number_of_loops_with_const_mean times to invoke
            stop.
        :param weighted_semivariance: (bool) if False then each distance is treated equally when calculating
            theoretical semivariance; if True then semivariances closer to the point have more weight,
        :param verbose: (bool) if True then all messages are printed, otherwise nothing.
        """

        # Class models
        self.experimental_semivariogram_of_areal_data = None  # Model updated in the point 1a
        self.initial_point_support_model = None  # Model updated in the point 2
        self.data_based_values = None  # Values updated in the point 2
        self.theoretically_regularized_model = None  # Values updated in the point 3
        self.optimal_point_support_model = None  # Values updated in the point 5
        self.optimal_regularized_model = None  # Values updated in the point 5
        self.rescalled_point_support_semivariogram = None  # Values updated in the point 6
        self.areal_semivariance_models = None  # Model updated in the step 8

        self.final_regularized = None
        self.final_optimal_point_support = None

        # Class parameters
        self.ranges = ranges
        self.weighted_semivariance = weighted_semivariance

        self.min_no_loops = min_no_loops
        if self.min_no_loops > 0:
            if loop_limit < self.min_no_loops:
                warning_loops = 'WARNING: Loop limit is smaller than minimum number of loops,' \
                                'minimum number of loops set to loop limit'
                print(warning_loops)
                self.min_no_loops = loop_limit
        else:
            self.min_no_loops = loop_limit

        self.loop_limit = loop_limit
        self.d_stat_change = d_statistics_change
        self.no_of_iters_const_mean = number_of_loops_with_const_mean  # Number of iterations with constant d-stat mean
        self.mean_vals_d_stat = []
        self.mean_diff = mean_diff

        self.sill_of_areal_data = None  # Value updated in the point 1b
        self.initial_deviation = None  # Value updated in the point 4
        self.optimal_deviation = None  # Value updated in the point 5
        self.deviations = []  # Value updated iteratively in the point 5
        self.weight_change = False  # Value changed in the main loop (points 6 to 8)
        self.weight = None  # Value updated in the rescale method (point 6 of the main function)
        self.weights = []  # Value updated iteratively in the rescale method (point 6 of the main function)
        self.iteration = 0  # Number of iterations
        self.deviation_change = 1

        # Print params
        self.verbose = verbose
        if self.verbose:
            self.msg = SemivariogramDeconvolutionMessages().msg

        self.class_name = "\nDeconvolution:"
        self.complete = "Process complete!"

    def _check_loops_status(self):
        """
        Method checks model loop's statistics to stop iteration after certain events occurs.
        """

        # Check loop limit
        if self.iteration < self.loop_limit:
            loop_limit = False
        else:
            loop_limit = True

        # Check minimum number of iterations
        if self.iteration < self.min_no_loops:
            min_loops = False
        else:
            min_loops = True

        # Check loop limit and min no of loops
        if loop_limit and min_loops:
            loop_arg = True
        else:
            loop_arg = False

        if self.verbose:
            print('')
            print('>> Iteration:', self.iteration)
            print('')
        return loop_arg

    def _check_optimizer(self):
        """
        Function checks if analysis is not in local optimum.
        """

        if len(self.deviations) % self.no_of_iters_const_mean == 0:
            d_stat_mean = np.mean(self.deviations[-self.no_of_iters_const_mean:])
            self.mean_vals_d_stat.append(d_stat_mean)
            if self.verbose:
                print('D stat mean:', d_stat_mean)
            if len(self.mean_vals_d_stat) > 1:
                if np.abs(self.mean_vals_d_stat[-1] - self.mean_vals_d_stat[-2]) < self.mean_diff:
                    if self.verbose:
                        print('Constant mean of deviation value, iteration stopped.', self.mean_diff)
                    return True
        else:
            return False

        return False

    def regularize_model(self, areal_data, areal_lags, areal_step_size,
                         areal_points_data, areal_points_lags, areal_points_step_size):
        """
        Method regularizes given areal model based on the:
        a) data with areal counts of some variable,
        b) data with population units and counts (divided per area),
        Based on the experimental semivariogram of areal centroids and population units function performs
        deconvolution and returns theoretical model for given areas.
        Method is described in: Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular
        Geographical Units, Mathematical Geology 40(1), 101-128, 2008.

        :param areal_data: (numpy array / list of lists)
            [area_id, area_geometry, centroid coordinate x, centroid coordinate y, value],
        :param areal_lags: (numpy array / list of lists) - array of lags (ranges of search),
        :param areal_step_size: (float) step size for search radius,
        :param areal_points_data: (numpy array / list of lists)
            [area_id, [point_position_x, point_position_y, value]]
        :param areal_points_lags: (numpy array / list of lists) - array of lags (ranges of search),
        :param areal_points_step_size: (float) step size for search radius.
        :return semivariance_models: (tuple) - (numpy array, TheoreticalSemivariance object) - function returns
            regularized semivariances as a numpy array [lag, semivariance value] and TheoreticalSemivariance model
            regularized and fitted into areal dataset.
        """

        # Setting up local variables from the while loop:
        regularized = []

        # 1a. Compute experimental semivariogram of areal data y_v_h

        if self.verbose:
            print(self.msg[1])
            print(self.msg[0])
            print(self.msg[-1])
            print(self.msg[0])
            print(self.msg[2])
            print(self.msg[0])

        areal_data_centroids = areal_data[:, 2:]
        experimental_semivariogram_areal = calculate_semivariance(areal_data_centroids,
                                                                  areal_lags, areal_step_size)

        self.experimental_semivariogram_of_areal_data = experimental_semivariogram_areal.copy()

        # 1b. Fit a model y_v_exp_h

        if self.verbose:
            print(self.msg[3])
            print(self.msg[0])

        theoretical_model = TheoreticalSemivariogram(areal_data_centroids,
                                                     self.experimental_semivariogram_of_areal_data,
                                                     verbose=self.verbose)

        theoretical_model.find_optimal_model(weighted=True, number_of_ranges=self.ranges)
        self.sill_of_areal_data = theoretical_model.params[1]

        # 2. Initial point support model definition y_0_h

        if self.verbose:
            print(self.msg[4])
            print(self.msg[0])

        self.initial_point_support_model = theoretical_model
        self.data_based_values = theoretical_model.predict(areal_lags)

        # 3. Model regularization procedure

        if self.verbose:
            print(self.msg[5])
            print(self.msg[0])

        # Initialize areal semivariance object
        areal_semivariance = ArealSemivariance(areal_data, areal_lags, areal_step_size,
                                               areal_points_data, areal_points_lags, areal_points_step_size,
                                               weighted_semivariance=self.weighted_semivariance)

        # Regularize semivariogram of areal data
        self.theoretically_regularized_model = areal_semivariance.regularize_semivariogram(
            empirical_semivariance=self.experimental_semivariogram_of_areal_data,
            theoretical_semivariance_model=self.initial_point_support_model)

        # 4. Quantify the deviation between data based experimental semivariogram
        # and theoretically regularized semivariogram

        if self.verbose:
            print(self.msg[6])
            print(self.msg[0])

        self.initial_deviation = self.calculate_deviation(self.theoretically_regularized_model[:, 1],
                                                          self.data_based_values)

        # 5. Setting up optimal models

        if self.verbose:
            print(self.msg[7])
            print(self.msg[0])

        self.optimal_point_support_model = self.initial_point_support_model
        self.optimal_regularized_model = self.theoretically_regularized_model
        self.optimal_deviation = self.initial_deviation
        self.deviations.append(self.optimal_deviation)

        loop_test = False

        while not loop_test:

            # 6. For each lag compute experimental values for the new point support semivariogram through a rescaling
            #    of the optimal point support model
            #    y(1)(h) = y_opt(h) x w(h)
            #    w(h) = 1 + [(y_exp_v(h) - y_opt_v(h) / (s^2 * sqrt(iter))]
            #    s = sill of the model y_exp_v
            #    iter = iteration number

            if self.verbose:
                print(self.msg[10])
                print(self.msg[0])

            self.rescalled_point_support_semivariogram = self.rescale()

            # 7. Fit a rescalled model using weighted least square regression (the same procedure as in step 1)

            if self.verbose:
                print(self.msg[11])
                print(self.msg[0])

            theoretical_model = TheoreticalSemivariogram(areal_data_centroids,
                                                         self.rescalled_point_support_semivariogram,
                                                         verbose=self.verbose)

            theoretical_model.find_optimal_model(weighted=True, number_of_ranges=self.ranges)
            temp_optimal_point_support_model = theoretical_model
            temp_sill_of_areal_data = theoretical_model.params[1]

            # 8. Regularize the model

            if self.verbose:
                print(self.msg[12])
                print(self.msg[0])

            areal_semivariance = ArealSemivariance(areal_data, areal_lags, areal_step_size,
                                                   areal_points_data, areal_points_lags, areal_points_step_size,
                                                   weighted_semivariance=self.weighted_semivariance)
            self.areal_semivariance_models = areal_semivariance

            regularized = areal_semivariance.regularize_semivariogram(
                empirical_semivariance=self.experimental_semivariogram_of_areal_data,
                theoretical_semivariance_model=temp_optimal_point_support_model)

            # 9. Compute the difference statistcs for the new model and decide what to do next

            if self.verbose:
                print(self.msg[13])
                print(self.msg[0])

            deviation = self.calculate_deviation(regularized[:, 1], self.data_based_values)
            self.deviations.append(deviation)

            if deviation < self.optimal_deviation:
                self.optimal_point_support_model = temp_optimal_point_support_model
                self.deviation_change = 1 - ((self.optimal_deviation - deviation) / self.optimal_deviation)
                self.optimal_deviation = deviation
                self.sill_of_areal_data = temp_sill_of_areal_data
                self.weight_change = False
            else:
                self.weight_change = True

            # Internal checking
            loop_test_loops = self._check_loops_status()
            loop_test_opt = self._check_optimizer()
            loop_test_d = self.deviation_change < self.d_stat_change
            loop_test = loop_test_loops or loop_test_opt or loop_test_d

        if self.verbose:
            print(self.msg[0])
            print(self.msg[-1])
            print(self.msg[20])
            print(self.msg[-1])
            print(self.msg[0])

        self.final_regularized = regularized
        self.final_optimal_point_support = self.optimal_point_support_model.predict(regularized[:, 0])

        semivariance_models = (self.final_regularized, self.optimal_point_support_model)
        return semivariance_models

    def rescale(self):
        """Function rescales the optimal point support model and creates new experimental values for
        each lag based on the equation:
            y(1)(h) = y_opt(h) x w(h)
            w(h) = 1 + [(y_exp_v(h) - y_opt_v(h) / (s^2 * sqrt(iter))]
            s = sill of the model y_exp_v
            iter = iteration number
        OUTPUT:
        :return rescalled_point_support_semivariogram: numpy array of the form [[lag, semivariance],
                                                                                [lag_x, semivariance_x],
                                                                                [..., ...]
                                                                               ]
        """

        self.iteration = self.iteration + 1
        y_opt_h = self.optimal_point_support_model.predict(self.optimal_regularized_model[:, 0])

        if not self.weight_change:
            i = np.sqrt(self.iteration)
            s = self.sill_of_areal_data
            c = s * i
            y_exp_v_h = self.data_based_values
            y_opt_v_h = self.optimal_regularized_model
            w = 1 + (y_exp_v_h - y_opt_v_h[:, 1]) / c
        else:
            w = 1 + ((self.weight - 1) / 2)

        self.weight = w.copy()
        self.weights.append(self.weight)

        if self.rescalled_point_support_semivariogram is None:
            rescalled = self.experimental_semivariogram_of_areal_data.copy()  # must have 3 dims
        else:
            rescalled = self.rescalled_point_support_semivariogram.copy()

        rescalled[:, 1] = y_opt_h * w

        return rescalled

    def calculate_deviation(self, regularized_model, data_based_model):
        """Function calculates deviation between experimental and theoretical semivariogram
            over given lags.
            INPUT:
            :param regularized_model: array of the values generated for the regularized model,
            :param data_based_model: array of modeled values generated from the theoretical model,
            OUTPUT:
            :return deviation: scalar which describes deviation between semivariograms.
        """

        data_based_model_len = len(data_based_model)

        if len(regularized_model) == data_based_model_len:
            if self.verbose:
                print('Start of deviation calculation')
            deviation = np.abs(regularized_model - data_based_model)
            deviation = np.divide(deviation,
                                  data_based_model,
                                  out=np.zeros_like(deviation),
                                  where=data_based_model != 0)
            deviation = sum(deviation) / data_based_model_len
            if self.verbose:
                print('Calculated deviation is:', deviation)
            return deviation
        else:
            raise ValueError('Length of data based model is different than length of regularized semivariogram')

    def export_regularized(self, filename):
        self.optimal_point_support_model.export_model(filename=filename)
        print('Model exported to file:', filename)

    def show_semivariograms(self):
        """
        Function shows experimental semivariogram, theoretical semivariogram and regularized semivariogram after
        semivariogram regularization.
        """
        plt.figure(figsize=(12, 12))
        plt.plot(self.experimental_semivariogram_of_areal_data[:, 0],
                 self.experimental_semivariogram_of_areal_data[:, 1], color='b')
        plt.plot(self.experimental_semivariogram_of_areal_data[:, 0],
                 self.final_optimal_point_support, color='r',
                 linestyle='--')
        plt.plot(self.final_regularized[:, 0], self.final_regularized[:, 1], color='g',
                 linestyle='-.')
        plt.legend(['Empirical semivariogram', 'Theoretical semivariogram',
                    'Regularized semivariogram, iteration {}'.format(self.iteration)])
        plt.title('Semivariograms comparison. Deviation value: {}'.format(self.optimal_deviation))
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    def show_deviation(self):
        """
        Function shows deviation of each algorithm loop after semivariogram regularization process.
        It could be used for testing purposes.
        """
        plt.figure(figsize=(12, 12))
        plt.plot(self.deviations)
        plt.title('Change of deviation over time')
        plt.xlabel('Iteration')
        plt.ylabel('Deviation')
        plt.show()
