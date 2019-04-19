# Base libraries
import numpy as np
import pandas as pd

# Data vizualization libraries
import matplotlib.pyplot as plt

# Pyinterpolate libraries
from pyinterpolate.kriging.semivariance_areal import ArealSemivariance
from pyinterpolate.kriging.semivariance_base import Semivariance
from pyinterpolate.kriging.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.kriging.helper_functions.euclidean_distance import centroid_blocks_distances


def calculate_semivariogram_deviation(data_based_semivariogram, theoretically_regularized_semivariogram):
    """Function calculates deviation between experimental and theoretical semivariogram
    over given lags.

    INPUT:
    :param data_based_semivariogram: data based semivariogram in the form of numpy array:
    [[lag 0, value 0],
     [lag i, value i],
     [lag z, value z]]

    :param theoretically_regularized_semivariogram: array in the same for as data_based_semivariogram,
    where first column represents the same lags as the first column of the data_based_semivariogram array.

    OUTPUT:
    :return deviation: scalar which describes deviation between semivariograms.
     """

    array_length = len(data_based_semivariogram)

    if array_length == len(theoretically_regularized_semivariogram):
        if (data_based_semivariogram[:, 0] == theoretically_regularized_semivariogram[:, 0]).any():
            print('Start of deviation calculation')
            deviation = np.abs(theoretically_regularized_semivariogram[:, 1] - data_based_semivariogram[:, 1])
            deviation = deviation / data_based_semivariogram[:, 1]
            deviation = sum(deviation) / array_length
            print('Calculated deviation is:', deviation)
            return deviation
        else:
            raise ValueError('Semivariograms have a different lags')
    else:
        raise ValueError('Length of data based semivariogram is different than length of theoretical semivariogram')


class DeconvolutedModel:
    """Class is a pipeline for deconvolution of areal data"""

    def __init__(self):

        # Models
        self.base_point_support = None
        self.experimental_point_support = None
        self.optimal_point_support_model = None
        self.optimal_regularized_model = None
        self.optimal_difference_statistics = None
        self.temp_point_support_model = None
        self.weights = 0

        # Loop parameters
        self.iteration_number = 0
        self.sill_of_areal_data = None
        self.difference_statistics_ratio = 1
        self.difference_statistics_limit = 0.01
        self.difference_statistics_decrease_number = 3

        # Areal Semivariance models
        self.within_block_semivariogram = None
        self.semivariogram_between_blocks = None

        # Additional
        self.centroid_distances = None
        self.areal_distances = None

    def regularize(self, areal_data_file, areal_lags, areal_step_size, data_column,
                   population_data_file, population_value_column, population_lags, population_step_size,
                   id_column_name):

        # Point support model
        centroids_semivar = Semivariance(areal_data_file, lags=areal_lags,
                                         step_size=areal_step_size,
                                         id_field=id_column_name)
        self.experimental_point_support = centroids_semivar.centroids_semivariance(data_column='LB RATES 2')
        self.centroid_distances = centroid_blocks_distances(centroids_semivar.g_dict)

        # Initial and optimal point support models

        theoretical_model = TheoreticalSemivariogram(centroids_semivar.centroids,
                                                     self.experimental_point_support)

        theoretical_model.find_optimal_model(weighted=True, number_of_ranges=100)

        self.optimal_point_support_model = theoretical_model
        self.sill_of_areal_data = theoretical_model.params[1]

        values_from_base = theoretical_model.calculate_values()
        self.base_point_support = values_from_base.T

        # Areal

        areal_semivariance = ArealSemivariance(self.optimal_point_support_model,
                                                areal_data_file, areal_lags, areal_step_size, data_column,
                                                population_data_file, population_value_column, population_lags,
                                                population_step_size, id_column_name)

        # Regularized Model
        regularized = areal_semivariance.blocks_semivariance()
        self.areal_distances = areal_semivariance.areal_distances
        self.within_block_semivariogram = areal_semivariance.within_block_semivariogram
        self.semivariogram_between_blocks = areal_semivariance.semivariogram_between_blocks
        self.optimal_regularized_model = regularized

        # Deviation
        deviation = calculate_semivariogram_deviation(self.experimental_point_support, regularized)
        self.optimal_difference_statistics = deviation

        # Model fitting loop
        while self.iteration_number < 20:
            self.iteration_number = self.iteration_number + 1

            # Rescale
            print('Rescaling procedure starts, iteration number: {}'.format(self.iteration_number))

            t = self.rescale()
            print('Model rescaled')
            print('Fitting of a rescalled model')

            print('Fit complete')
            print('Regularization of the rescalled model')

            print('Regularization complete')
            print('Difference statistic computation')
            


        return self.optimal_point_support_model, self.optimal_regularized_model

    def rescale(self):
        """Function rescales the optimal point support model and creates new experimental values for
        each lag based on the equation:
        gamma{1}(h) = gamma{opt}(h) x w{1}(h) with:
        w{1}(h) = 1 + [(gamma{exp-v}(h) - gamma{opt-v}(h)) / (s{exp}^2 * sqrt(iter)),
        s{exp}^2 - sill of the point support model,
        iter - number of iteration of the algorithm

        OUTPUT:
        :return rescalled_point_support_semivariogram: numpy array of the form [[lag, semivariance],
                                                                                [lag_x, semivariance_x],
                                                                                [..., ...]
                                                                               ]
        """

        y_opt = self.base_point_support
        s = self.sill_of_areal_data**2
        y_v_opt = self.optimal_regularized_model
        i = self.iteration_number

        w = 1 + (y_opt - y_v_opt[:, 1]) / (s * np.sqrt(i))
        self.weights = w.copy()
        rescalled = y_v_opt.copy()
        rescalled[:, 1] = y_opt * w
        self.temp_point_support_model = rescalled

        return rescalled

    # Data visualization methods

    def show_distances(self, centroids=None, weights=None):

        if not centroids:
            x = pd.DataFrame.from_dict(self.centroid_distances, orient='index')
            x.sort_index(inplace=True)
            x.sort_index(axis=1, inplace=True)
            x = (x.values).ravel()
            centroids = np.where(x > 0, np.log(x), 0)
            centroids[centroids == 0] = np.nan

        if not weights:
            y = pd.DataFrame.from_dict(self.areal_distances, orient='index')
            y.sort_index(inplace=True)
            y.sort_index(axis=1, inplace=True)
            y = (y.values).ravel()
            weights = np.where(y > 0, np.log(y), 0)
            weights[weights == 0] = np.nan

        plt.figure(figsize=(12, 12))
        plt.scatter(x=centroids, y=weights, s=3, alpha=0.8)
        plt.title("Population-weighted distances")
        plt.xlabel("Log10 (block distance)")
        plt.ylabel("Log10 (Euclidean distance)")
        plt.plot()

    def show_semivariograms(self):
        plt.figure(figsize=(12, 12))
        plt.plot(self.experimental_point_support[:, 0], self.experimental_point_support[:, 1], color='b')
        plt.plot(self.experimental_point_support[:, 0], self.base_point_support, color='r',
                 linestyle='--')
        plt.plot(self.optimal_regularized_model[:, 0], self.optimal_regularized_model[:, 1], color='g',
                 linestyle='-.')
        plt.legend(['Empirical semivariogram', 'Theoretical semivariogram',
                    'Regularized semivariogram, iteration {}'.format(self.iteration_number)])
        plt.title('Semivariograms comparison')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()
