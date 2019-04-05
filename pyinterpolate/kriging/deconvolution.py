# Pyinterpolate libraries
from pyinterpolate.kriging.model_regularization import calculate_semivariogram_deviation
from pyinterpolate.kriging.semivariance_areal import ArealSemivariance
from pyinterpolate.kriging.semivariance_base import Semivariance


class DeconvolutedModel:
    """Class is a pipeline for deconvolution of areal data"""

    def __init__(self):
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


    def regularize(self, areal_data_file, areal_lags, areal_step_size, data_column,
                 population_data_file, population_value_column, population_lags, population_step_size,
                 id_column_name):

        areal_semivariance = ArealSemivariance(areal_data_file, areal_lags, areal_step_size, data_column,
                 population_data_file, population_value_column, population_lags, population_step_size,
                 id_column_name)

        