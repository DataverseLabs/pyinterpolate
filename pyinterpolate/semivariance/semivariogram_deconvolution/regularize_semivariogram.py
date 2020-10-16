# Base libraries
import numpy as np

# Data vizualization libraries
import matplotlib.pyplot as plt

# Pyinterpolate libraries
from pyinterpolate.semivariance.areal_semivariance.areal_semivariance import ArealSemivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import \
    calculate_semivariance  # Experimental semivariogram


class RegularizedSemivariogram:
    """
    Class performs deconvolution of semivariogram of areal data. Whole procedure is based on the iterative process
    described in: Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical
    Units, Mathematical Geology 40(1), 101-128, 2008.

    Class works as follow:

    - initialize your object (no parameters),
    - then use fit() method to build initial point support model,
    - the use transform() method to perform semivariogram regularization.

    Class public methods:

    fit() - fits areal data and point support data into a model, initialize experimental semivariogram,
    theoretical semivariogram model, regularized point support model and deviation.

    transform() - performs semivariogram regularization, which is an iterative process.

    show_semivariograms() - plots experimental semivariogram of areal data, theoretical curve of areal data,
    regularized model values and regularized model theoretical curve.
    """

    def __init__(self):
        """
        Class has multiple params, some of them are designed to control process of regularization and other are storing
        semivariogram models and experimental (or regularized) values of semivariance.
        """

        # Procedure control parameters
        self.iter = 0
        self.max_iters = None

        self.deviation_ratio = 1
        self.min_deviation_ratio = None

        self.diff_decrease = 1
        self.min_diff_decrease = None

        self.const_d_stat_reps = 0
        self.min_diff_decrease_reps = None

        self.weight_error_lags = False

        self.weight_change = False

        self.store_models = False

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
        self.temp_experimental_semivariogram = None
        self.temp_theoretical_semivariogram_model = None
        self.temp_regularized_model = None
        self.temp_deviation = None

        # Optimal semivariogram models and params
        self.optimal_theoretical_model = None
        self.optimal_regularized_model = None
        self.optimal_deviation = None

        # Final models
        self.final_theoretical_model = None
        self.final_optimal_model = None

        # Data
        self.areal_data = None
        self.areal_lags = None
        self.areal_step_size = None
        self.point_support_data = None

        # Stored models if self.store_models is True
        self.t_exp_list = []
        self.t_theo_list = []
        self.t_reg_list = []

    def _regularize(self, empirical_semivariance, semivariance_model):
        """
        Function regularizes semivariogram with ArealSemivariance class.

        INPUT:

        :param empirical_semivariance: experimental values of semivariance as an array of the form:
            [[column with lags, column with values, column with number of points within lag]],
        :param semivariance_model: TheoreticalSemivariance model,

        OUTPUT:

        :return: regularized semivariance values (array).
        """

        # Initialize areal semivariance object
        areal_semivariance = ArealSemivariance(self.areal_data,
                                               self.areal_lags,
                                               self.areal_step_size,
                                               self.point_support_data,
                                               weighted_semivariance=self.weight_error_lags)

        # Regularize semivariogram of areal data
        theoretically_regularized_model_values = areal_semivariance.regularize_semivariogram(
            empirical_semivariance=empirical_semivariance,
            theoretical_semivariance_model=semivariance_model)

        return theoretically_regularized_model_values[:, 1]

    def _calculate_deviation(self, regularized_model, theoretical_model):
        """
        Function calculates deviation between experimental and theoretical semivariogram over given lags.

        INPUT:

        :param regularized_model: (numpy array) array of the values generated for the regularized model,
        :param theoretical_model: (TheoreticalSemivariance) theoretical model of data,

        OUTPUT:

        :return deviation: (float) scalar which describes deviation between semivariograms.
        """

        lags = self.areal_lags
        theoretical_values = theoretical_model.predict(lags)
        regularized_values = regularized_model

        deviation = np.abs(regularized_values - theoretical_values)
        deviation = np.divide(deviation,
                              theoretical_values,
                              out=np.zeros_like(deviation),
                              where=theoretical_values != 0)
        deviation = np.mean(deviation)
        return deviation

    def _rescale_optimal_point_support(self):
        """Function rescales the optimal point support model and creates new experimental values for each lag based on
            the equation:

            y(1)(h) = y_opt(h) x w(h)

            w(h) = 1 + [(y_exp_v(h) - y_opt_v(h) / (s^2 * sqrt(iter))]

            where:

            - s = sill of the model y_exp_v
            - iter = iteration number

        OUTPUT:

        :return rescalled_point_support_semivariogram: (numpy array) of the form [[lag, semivariance, number of points]]
        """
        lags = self.areal_lags

        y_opt_h = self.optimal_theoretical_model.predict(lags)

        if not self.weight_change:
            sill = self.initial_theoretical_model_of_areal_data.params[1]
            denominator = sill * np.sqrt(self.iter)

            y_exp_v_h = self.initial_theoretical_model_of_areal_data.predict(lags)
            y_opt_v_h = self.optimal_regularized_model

            numerator = (y_exp_v_h - y_opt_v_h)

            w = 1 + (numerator / denominator)
        else:
            w = 1 + ((self.weights[-1] - 1) / 2)

        rescalled = self.experimental_semivariogram_of_areal_data.copy()
        rescalled[:, 1] = y_opt_h * w

        return rescalled, w

    def _check_deviation_ratio(self):
        return bool(self.deviation_ratio <= self.min_deviation_ratio)

    def _check_loop_limit(self):
        return bool(self.iter >= self.max_iters)

    def _check_diff_d_stat(self):
        if self.diff_decrease < self.min_diff_decrease:

            if self.const_d_stat_reps >= self.min_diff_decrease_reps:
                return True

            self.const_d_stat_reps += 1
            return False

        if self.const_d_stat_reps >= 1:

            self.const_d_stat_reps = self.const_d_stat_reps - 1
            return False

        return False

    def _check_algorithm(self):
        t1 = self._check_deviation_ratio()  # Default False
        t2 = self._check_diff_d_stat()  # Default False
        t3 = self._check_loop_limit()  # Default False

        cond = not (t1 or t2 or t3)  # Default False

        return cond

    def fit(self, areal_data, areal_lags, areal_step_size,
            point_support_data, ranges=16, weighted_lags=True, store_models=False):
        """
        Function fits areal and point support data to the initial regularized models.

        INPUT:

        :param areal_data: areal data prepared with the function prepare_areal_shapefile(), where data is a numpy array
            in the form: [area_id, area_geometry, centroid coordinate x, centroid coordinate y, value],
        :param areal_lags: list of lags between each distance,
        :param areal_step_size: step size between each lag, usually it is a half of distance between lags,
        :param point_support_data: point support data prepared with the function get_points_within_area(), where data is
            a numpy array in the form: [area_id, [point_position_x, point_position_y, value]],
        :param ranges: (int) number of ranges to test during semivariogram fitting. More steps == more accurate nugget
            and range prediction, but longer calculations,
        :param weighted_lags: (bool) lags weighted by number of points; if True then during semivariogram fitting error
            of each model is weighted by number of points for each lag. In practice it means that more reliable data
            (lags) have larger weights and semivariogram is modeled to better fit to those lags,
        :param store_models: (bool) if True then experimental, regularized and theoretical models are stored in lists
            after each iteration. It is important for a debugging process.
        """

        # Update data class params
        self.areal_data = areal_data
        self.areal_lags = areal_lags
        self.areal_step_size = areal_step_size
        self.point_support_data = point_support_data
        self.ranges = ranges
        self.weight_error_lags = weighted_lags

        self.store_models = store_models

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

    def transform(self, max_iters=25, min_deviation_ratio=0.01, min_diff_decrease=0.01, min_diff_decrease_reps=3):
        """
        Function transofrms fitted data and performs semivariogram regularziation iterative procedure.

        INPUT:

        :param max_iters: maximum number of iterations,
        :param min_deviation_ratio: minimum ration between deviation and initial deviation (D(i) / D(0)) below each
            algorithm is stopped,
        :param min_diff_decrease: minimum absolute difference between new and optimal deviation divided by optimal
            deviation: ABS(D(i) - D(opt)) / D(opt). If it is recorded n times (controled by the min_diff_d_stat_reps
            param) then algorithm is stopped,
        :param min_diff_decrease_reps: (int) number of iterations when algorithm is stopped if condition
            min_diff_d_stat is fulfilled.
        """

        # Check if data is fitted
        if self.initial_regularized_model is None:
            raise RuntimeError('Before transform you must fit areal data and calculate initial point support models')

        # Update class control params
        self.iter = 1
        self.max_iters = max_iters

        self.min_deviation_ratio = min_deviation_ratio

        self.min_diff_decrease = min_diff_decrease
        self.min_diff_decrease_reps = min_diff_decrease_reps

        # Update initial optimal models
        self.optimal_theoretical_model = self.initial_theoretical_model_of_areal_data
        self.optimal_regularized_model = self.initial_regularized_model
        self.optimal_deviation = self.initial_deviation

        # Prepare semivariogram modeling data
        areal_centroids = self.areal_data[:, 2:]
        ranges = self.ranges
        is_weighted = self.weight_error_lags

        # Append initial models if self.store_models is True

        if self.store_models:
            self.t_theo_list.append(self.optimal_theoretical_model)
            self.t_reg_list.append(self.optimal_regularized_model)
            self.t_exp_list.append(self.experimental_semivariogram_of_areal_data)

        # Start iteration procedure

        while self._check_algorithm():
            # Compute new experimental values for new experimental point support model

            self.temp_experimental_semivariogram, weights = self._rescale_optimal_point_support()
            self.weights.append(weights)

            # Fit rescaled empirical semivariogram to the new theoretical function
            self.temp_theoretical_semivariogram_model = TheoreticalSemivariogram(areal_centroids,
                                                                                 self.temp_experimental_semivariogram)
            self.temp_theoretical_semivariogram_model.find_optimal_model(
                weighted=is_weighted,
                number_of_ranges=ranges
            )

            # Regularize model
            self.temp_regularized_model = self._regularize(
                self.temp_experimental_semivariogram,
                self.temp_theoretical_semivariogram_model
            )

            # Compute difference statistics

            self.temp_deviation = self._calculate_deviation(self.temp_regularized_model,
                                                            self.initial_theoretical_model_of_areal_data)

            if self.temp_deviation < self.optimal_deviation:
                self.weight_change = False

                self.diff_decrease = np.abs(self.temp_deviation - self.optimal_deviation) / self.optimal_deviation
                self.deviation_ratio = self.temp_deviation / self.deviations[0]

                self.optimal_deviation = self.temp_deviation

                # Update models
                self.optimal_theoretical_model = self.temp_theoretical_semivariogram_model
                self.optimal_regularized_model = self.temp_regularized_model

            else:
                self.weight_change = True

            self.deviations.append(self.temp_deviation)
            self.iter = self.iter + 1

            # Update models if self.store_models is set to True
            if self.store_models:
                self.t_theo_list.append(self.temp_theoretical_semivariogram_model)
                self.t_exp_list.append(self.temp_experimental_semivariogram)
                self.t_reg_list.append(self.temp_regularized_model)

        # Get theoretical model from regularized
        self.final_theoretical_model = self.temp_theoretical_semivariogram_model
        self.final_optimal_model = self.optimal_regularized_model

    def export_regularized_model(self, filename):
        """
        Function exports final regularized model parameters into specified csv file.
        """
        
        if self.final_theoretical_model is None:
            raise RuntimeError('You cannot export any model if you not transform data.')
        
        self.final_theoretical_model.export_model(filename)

    def show_baseline_semivariograms(self):
        """
        Function shows experimental semivariogram, initial theoretical semivariogram and
        initial regularized semivariogram after fit() operation.
        """
        lags = self.experimental_semivariogram_of_areal_data[:, 0]
        plt.figure(figsize=(12, 12))
        plt.plot(lags, self.experimental_semivariogram_of_areal_data[:, 1], color='b')
        plt.plot(lags, self.initial_theoretical_model_of_areal_data.predict(lags), color='r', linestyle='--')
        plt.plot(lags, self.initial_regularized_model, color='g', linestyle='-.')
        plt.legend(['Experimental semivariogram of areal data', 'Initial Semivariogram of areal data',
                    'Regularized data points'])
        plt.title('Semivariograms comparison. Deviation value: {}'.format(self.initial_deviation))
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

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
                 self.initial_theoretical_model_of_areal_data.predict(lags), color='r',
                 linestyle='--')
        plt.plot(lags, self.optimal_regularized_model, color='g',
                 linestyle='-.')
        plt.plot(lags,
                 self.optimal_theoretical_model.predict(lags), color='black', linestyle='dotted')
        plt.legend(['Experimental semivariogram of areal data', 'Initial Semivariogram of areal data',
                    'Regularized data points, iteration {}'.format(self.iter),
                    'Optimized theoretical point support model'])
        plt.title('Semivariograms comparison. Deviation value: {}'.format(self.optimal_deviation))
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()
