# Base libraries
import numpy as np

# Data vizualization libraries
import matplotlib.pyplot as plt

# Pyinterpolate libraries
from pyinterpolate.kriging.semivariance_areal import ArealSemivariance
from pyinterpolate.kriging.semivariance_base import Semivariance
from pyinterpolate.kriging.fit_semivariance import TheoreticalSemivariogram


class RegularizedModel:
    """Class serves as the deconvolution model for an areal data"""

    def __init__(self, scaling_factor=2,
                 d_statistics_change=0.1,
                 ranges=32,
                 loop_limit=20,
                 min_no_loops=0,
                 number_of_loops_with_const_mean=5,
                 mean_diff=0.01):

        # Class models
        self.experimental_semivariogram_of_areal_data = None  # Model updated in the point 1a
        self.initial_point_support_model = None  # Model updated in the point 2
        self.data_based_values = None  # Values updated in the point 2
        self.theoretically_regularized_model = None  # Values updated in the point 3
        self.optimal_point_support_model = None  # Values updated in the point 5
        self.optimal_regularized_model = None  # Values updated in the point 5
        self.rescalled_point_support_semivariogram = None  # Values updated in the point 6

        self.final_regularized = None
        self.final_optimal_point_support = None

        # Class parameters
        self.ranges = ranges
        self.scaling_factor = scaling_factor

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
        self.class_name = "\nDeconvolution:"
        self.complete = "Process complete!"

    def _check_loops_status(self):
        """Method checks model loop's statistics to stop iteration after certain events occurs"""

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

        print('')
        print('>> Iteration:', self.iteration)
        print('')
        return loop_arg

    def _check_optimizer(self):
        # Function checks if analysis is not in local optimum

        if len(self.deviations) % self.no_of_iters_const_mean == 0:
            d_stat_mean = np.mean(self.deviations[-self.no_of_iters_const_mean:])
            self.mean_vals_d_stat.append(d_stat_mean)
            print('D stat mean:', d_stat_mean)
            if len(self.mean_vals_d_stat) > 1:
                if np.abs(self.mean_vals_d_stat[-1] - self.mean_vals_d_stat[-2]) < self.mean_diff:
                    print('Constant mean of deviation value, iteration stopped.', self.mean_diff)
                    return True
        else:
            return False

        return False

    def regularize_model(self, areal_data_file, areal_lags, areal_step_size, data_column,
                         population_data_file, population_value_column, population_lags, population_step_size,
                         id_column_name):

        """Method regularizes given areal model based on the:
        a) data with areal counts of some variable,
        b) data with population units and counts (divided per area),
        Based on the experimental semivariogram of areal centroids and population units function performs
        deconvolution and returns theoretical model for given areas.
        Method is described in: Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular
        Geographical Units, Mathematical Geology 40(1), 101-128, 2008.
        """

        # Setting up local variables from the while loop:
        regularized = []

        # 1a. Compute experimental semivariogram of areal data y_v_h

        print(self.class_name)
        print('Computation of experimental semivariogram of areal data...')

        semivar = Semivariance(areal_data_file,
                               lags=areal_lags,
                               step_size=areal_step_size,
                               id_field=id_column_name
                               )
        self.experimental_semivariogram_of_areal_data = semivar.centroids_semivariance(data_column=data_column)
        print(self.complete)

        # 1b. Fit a model y_v_exp_h

        print(self.class_name)
        print('Fitting theoretical model to the areal data')

        centroids = semivar.centroids.copy()

        theoretical_model = TheoreticalSemivariogram(centroids[:, :3],
                                                     self.experimental_semivariogram_of_areal_data)

        theoretical_model.find_optimal_model(weighted=True, number_of_ranges=self.ranges)
        self.sill_of_areal_data = theoretical_model.params[1]

        print(self.complete)

        # 2. Initial point support model definition y_0_h\

        print(self.class_name)
        print('Setting of the initial point support model - function and parameters')

        self.initial_point_support_model = theoretical_model
        self.data_based_values = (theoretical_model.calculate_values()).T

        print(self.complete)

        # 3. Model regularization procedure

        print(self.class_name)
        print('Areal Semivariance fitting to the initial point support model')

        areal_semivariance = ArealSemivariance(self.initial_point_support_model,
                                               areal_data_file, areal_lags, areal_step_size, data_column,
                                               population_data_file, population_value_column, population_lags,
                                               population_step_size, id_column_name)

        self.theoretically_regularized_model = areal_semivariance.blocks_semivariance()
        print(self.complete)

        # 4. Quantify the deviation between data based experimental semivariogram
        # and theoretically regularized semivariogram

        print(self.class_name)
        print('Deviation estimation')

        self.initial_deviation = self.calculate_deviation(self.theoretically_regularized_model[:, 1],
                                                          self.data_based_values)

        print(self.complete)

        # 5. Setting up optimal models

        print(self.class_name)
        print('Setting up optimal models')

        self.optimal_point_support_model = self.initial_point_support_model
        self.optimal_regularized_model = self.theoretically_regularized_model
        self.optimal_deviation = self.initial_deviation
        self.deviations.append(self.optimal_deviation)

        print(self.complete)

        loop_test = False
        while not loop_test:

            # 6. For each lag compute experimental values for the new point support semivariogram through a rescaling
            #    of the optimal point support model
            #    y(1)(h) = y_opt(h) x w(h)
            #    w(h) = 1 + [(y_exp_v(h) - y_opt_v(h) / (s^2 * sqrt(iter))]
            #    s = sill of the model y_exp_v
            #    iter = iteration number

            print(self.class_name)
            print('Beginning of rescalling procedure')

            self.rescalled_point_support_semivariogram = self.rescale()

            print(self.complete)

            # 7. Fit a rescalled model using weighted least square regression (the same procedure as in step 1)

            print(self.class_name)
            print('Computation of experimental semivariogram of rescalled data...')

            theoretical_model = TheoreticalSemivariogram(centroids[:, :3],
                                                         self.rescalled_point_support_semivariogram)

            theoretical_model.find_optimal_model(weighted=True, number_of_ranges=self.ranges)
            temp_optimal_point_support_model = theoretical_model
            temp_sill_of_areal_data = theoretical_model.params[1]

            print(self.complete)

            # 8. Regularize the model

            print(self.class_name)
            print('Regularization of the rescalled model')

            areal_semivariance = ArealSemivariance(temp_optimal_point_support_model,
                                                   areal_data_file, areal_lags, areal_step_size, data_column,
                                                   population_data_file, population_value_column, population_lags,
                                                   population_step_size, id_column_name)

            # Regularized Model
            regularized = areal_semivariance.blocks_semivariance()
            plt.figure()
            plt.plot(regularized[:, 0], regularized[:, 1])
            plt.show()

            print(self.complete)

            # 9. Compute the difference statistcs for the new model and decide what to do next

            print(self.class_name)
            print('Difference statistics calculation...')

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

            print(self.complete)

            # Internal checking
            loop_test_loops = self._check_loops_status()
            loop_test_opt = self._check_optimizer()
            loop_test_d = self.deviation_change < self.d_stat_change
            print(loop_test_loops, loop_test_opt, loop_test_d)
            loop_test = loop_test_loops or loop_test_opt or loop_test_d

        print(self.class_name)
        print('\n')
        print('##################')
        print('Setting final models')
        print('##################')

        self.final_regularized = regularized
        self.final_optimal_point_support = self.optimal_point_support_model.predict(regularized[:, 0])

        print('Models set')
        print(self.complete)

        return self.optimal_point_support_model

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

    @staticmethod
    def calculate_deviation(regularized_model, data_based_model):
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
            print('Start of deviation calculation')
            deviation = np.abs(regularized_model - data_based_model)
            deviation = np.divide(deviation,
                                  data_based_model,
                                  out=np.zeros_like(deviation),
                                  where=data_based_model != 0)
            deviation = sum(deviation) / data_based_model_len
            print('Calculated deviation is:', deviation)
            return deviation
        else:
            raise ValueError('Length of data based model is different than length of regularized semivariogram')

    def show_semivariograms(self):
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
