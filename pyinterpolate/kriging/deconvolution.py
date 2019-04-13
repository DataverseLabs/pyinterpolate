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
        self.experimental_point_support = None
        self.initial_point_support_model = None
        self.optimal_point_support_model = None
        self.optimal_regularized_model = None
        self.optimal_difference_statistics = None
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

        areal_semivariance = ArealSemivariance(areal_data_file, areal_lags, areal_step_size, data_column,
                 population_data_file, population_value_column, population_lags, population_step_size,
                 id_column_name)
        
        # Point support model
        centroids_semivar = Semivariance(areal_data_file, lags=areal_lags,
                                          step_size=areal_step_size,
                                          id_field=id_column_name)
        self.experimental_point_support = centroids_semivar.centroids_semivariance(data_column='LB RATES 2')
        self.centroid_distances = centroid_blocks_distances(centroids_semivar.g_dict)
        
        # Initial and optimal point support models
        
        theoretical_model = TheoreticalSemivariogram(centroids_semivar.centroids,
                                                    self.experimental_point_support)
        optimal_model = theoretical_model.find_optimal_model(weighted=True, number_of_ranges=100)
        
        self.optimal_point_support_model = theoretical_model
        
        # Regularized Model
        regularized = areal_semivariance.blocks_semivariance()
        self.areal_distances = areal_semivariance.areal_distances
        self.within_block_semivariogram = areal_semivariance.within_block_semivariogram
        self.semivariogram_between_blocks = areal_semivariance.semivariogram_between_blocks
        self.optimal_regularized_model = regularized
        
        # Deviation
        deviation = calculate_semivariogram_deviation(self.experimental_point_support, regularized)
        self.optimal_difference_statistics = deviation
        
        return self.optimal_point_support_model, self.optimal_regularized_model
    
    
    # Data visualization methods
    
    def show_distances(self, centroids=None, weights=None):
        
        if not centroids:
            x = pd.DataFrame.from_dict(self.centroid_distances, orient='index')
            x.sort_index(inplace=True)
            x.sort_index(axis=1, inplace=True)
            x = (x.values).ravel()
            centroids = np.where(x>0, np.log(x), 0)
            centroids[centroids == 0] = np.nan
            
        if not weights:
            y = pd.DataFrame.from_dict(self.areal_distances, orient='index')
            y.sort_index(inplace=True)
            y.sort_index(axis=1, inplace=True)
            y = (y.values).ravel()
            weights = np.where(y>0, np.log(y), 0)
            weights[weights == 0] = np.nan
        
        plt.figure(figsize=(12, 12))
        plt.scatter(x=centroids, y=weights, s=3, alpha=0.8)
        plt.title("Population-weighted distances")
        plt.xlabel("Log10 (block distance)")
        plt.ylabel("Log10 (Euclidean distance)")
        plt.plot()