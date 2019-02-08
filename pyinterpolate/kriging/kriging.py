import sys

import math
import numpy as np
import matplotlib.pyplot as plt
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance


class Krige:
    """
    Class for kriging interpolation of the unknown values in a given location (position). Class takes two arguments
    during the initialization:
    data - list of known points and their values,
    semivariogram_model - semivariogram model calculated with the TheoreticalSemivariogram class.
    
    Available methods:
    * prepare_data(unknown_position, number_of_records, verbose) - method prepares data and it is necessary to run it before
    the kriging interpolation,
    * ordinary_kriging(test_for_anomalies=True) - method performs ordinary kriging interpolation on the prepared dataset of
    unknown points and if parameter test_for_anomalies is set to True then method monitor output for the negative values and
    inform user about it when negative value occur,
    * simple_kriging(area_mean=None, test_for_anomalies=True) - method performs simple kriging interpolation 
    on the prepared dataset of unknown points and if parameter test_for_anomalies is set to True then method monitor 
    output for the negative values and sinform user about it when negative values occur,
    * normalize_weights(weights, estimated_value) - method for negative weights removal from the weight matrix.
    * interpolate_raster(scale_factor=0.01, kriging_type='ordinary', number_of_neighbours=4, update_model=False) - method
    for interpolation of a raster based on the grid limited by the extreme points from the known values, it may be used
    instead of kriging interpolation methods for fast visualization of the data over a surface,
    * create_raster(scale_factor, points_list=None) - method creates canvas (numpy array) for interpolation.
    
    Static methods:
    * prepare_min_distance(rng, sf) - helper method for create_raster method. It is used to find limits of the canvas.
    
    Data visualization methods:
    * show_known_points() - method shows scatterplot of known points locations,
    * show_results(values_matrix=None) - method shows interpolated surface. If numpy array matrix is not given then method
    takes interpolated raster from the class instance (it must be calculated with the interpolate_raster method),
    * show_error_matrix(error_matrix=None) - method shows interpolated error surface. If numpy array matrix is not 
    given then method takes error raster from the class instance (it must be calculated with the interpolate_raster method).
    
    Example how to prepare model:
    
    model = Krige(data, semivariogram)
    model.interpolate_raster(scale_factor=0.01,
                         kriging_type='ordinary',
                         number_of_neighbours=10,
                         update_model=True)
    model.show_results()
    model.show_error_matrix()
    
    """

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
        self.distances = None
        self.interpolated_raster = None
        self.interpolated_error_matrix = None

    def prepare_data(self, unknown_position, number_of_records=10, verbose=False):
        """
        :param unknown_position: array with position of unknown value
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
        self.prepared_data = np.array(output_data)
        if verbose:
            print(('Point of position {} prepared for processing').format(
                unknown_position))
            
    def normalize_weights(self, weights, estimated_value, kriging_type):
        """
        Algorithm for weight normalization to remove negative weights of the points which are
        clustering. Derived from Deutsch, C.V., Correcting for negative weights in
        ordinary kriging, Computers & Geosciences, Vol. 22, No. 7, pp. 765-773, 1996.
        
        :param: weights - weights matrix calculated with "normal" kriging procedure,
        :param: estimated_value - value estimated for a given, unknown point.
        :return: weight_matrix - normalized weight matrix where negative weights are removed and
        matrix is scalled to give a sum of all elements equal to 1.
        """
        
        if kriging_type == 'ord':
            weight_matrix = weights[:-1].copy()
            output_matrix = weights[:-1].copy()
        elif kriging_type == 'sim':
            weight_matrix = weights.copy()
            output_matrix = weights.copy()
        else:
            print('You did not choose any kriging type. Chosen type: <sim> - simple kriging.')
            weight_matrix = weights.copy()
            output_matrix = weights.copy()
        
        ###### Calculate average covariance between the location being ######
        ###### estimated and the locations with negative weights       ######
        
        locs = np.argwhere(weight_matrix < 0)  # Check where weights are below 0.0
        locs = locs[:, 0]
        
        # Calculate covariance between those points and unknown point
        if len(locs) >= 1:
            C = []
            mu = 0
            for i in locs:
                c = estimated_value * self.prepared_data[i, 2]
                mu = mu + estimated_value + self.prepared_data[i, 2]
                C.append(c)
                output_matrix[i, 0] = 0
            mu = mu / len(C)
            cov = np.sum(C) / len(C) - mu*mu

            ###### Calculate absolute magnitude of the negative weights #####

            w = weight_matrix[weight_matrix < 0]
            w = w.T
            magnitude = np.sum(np.abs(w)) / len(w)

            ###### Test values greater than 0 and check if they need to be
            ###### rescaled to 0 ######

            ###### if weight > 0 and Covariance between unknown point and known
            ###### point is less than the average covariance between the location
            ###### being estimated and the locations with negative weights and
            ###### and weight is less than absolute magnitude of the negative 
            ###### weights then set weight to zero #####

            positive_locs = np.argwhere(weight_matrix > 0)  # Check where weights are greater than 0.0
            positive_locs = positive_locs[:, 0]

            for j in positive_locs:
                cov_est = (estimated_value * self.prepared_data[j, 2]) / 2
                mu = (estimated_value + self.prepared_data[j, 2]) / 2
                cov_est = cov_est - mu*mu
                if cov_est < cov:
                    if weight_matrix[j, 0] < magnitude:
                        output_matrix[j, 0] = 0

            ###### Normalize weight matrix to get a sum of all elements equal to 1 ######

            output_matrix = output_matrix / np.sum(output_matrix)

            return output_matrix
        else:
            return weights
        

    def ordinary_kriging(self, test_for_anomalies=True):
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
        
        # Test for negative values
        if test_for_anomalies:
            if zhat < 0:
                user_input_message = 'Estimated value is below zero and it is set to: {}. Continue? \
                (Type <y> if yes or <n> if no)\n'.format(zhat)
                if user_input_message is 'y':
                    print('Program is running...')
                else:
                    sys.exit('Program is terminated. Try different semivariogram model. (Did you use gaussian model? \
                             If so then try to use simpler models like linear or exponential) and/or analyze your data \
                             for any clusters which may affect the final estimation')
                    
        estimated_weights = self.normalize_weights(w, zhat, 'ord')
        zhat = (np.matrix(self.prepared_data[:, -2] * w[:-1])[0, 0])
        sigmasq = (w.T * k)[0, 0]
        if sigmasq < 0:
            sigma = 0
        else:
            sigma = np.sqrt(sigmasq)
        return zhat, sigma, w[-1][0], w

    def simple_kriging(self, area_mean=None, test_for_anomalies=True):
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
        predicted = self.model.predict(distances.ravel())
        predicted = np.matrix(predicted.reshape(n, n))

        w = np.linalg.solve(predicted, k)
        r = self.prepared_data[:, -2] - area_mean
        zhat = (np.matrix(r) * w)[0, 0]
        zhat += area_mean
        
        # Test for negative values
        if test_for_anomalies:
            if zhat < 0:
                user_input_message = 'Estimated value is below zero and it is set to: {}. Continue? \
                (Type <y> if yes or <n> if no)\n'.format(zhat)
                if user_input_message is 'y':
                    print('Program is running...')
                else:
                    sys.exit('Program is terminated. Try different semivariogram model. (Did you use gaussian model? \
                             If so then try to use simpler models like linear or exponential) and/or analyze your data \
                             for any clusters which may affect the final estimation')
        
        w = self.normalize_weights(w, zhat, 'sim')
        zhat = (np.matrix(r) * w)[0, 0]
        zhat += area_mean
        sigmasq = (w.T * k)[0, 0]
        if sigmasq < 0:
            sigma = 0
        else:
            sigma = np.sqrt(sigmasq)

        return zhat, sigma, area_mean, w
    
    @staticmethod
    def prepare_min_distance(rngs, sf):
        if rngs[0] < rngs[1]:
            return sf * rngs[0]
        else:
            return sf * rngs[1]

    def create_raster(self, scale_factor, points_list=None):
        """
        Method creates canvas raster for interpolated values. Canvas is created based on the edge points
        of the known area.
        :param scale_factor: Scale factor is multiplied by a smaller distance from width or height of
        a raster. This gives one unit (pixel width / height) of a created map. Scale factor must be a fraction.
        :param points_list: Points with known values (x, y)
        :return: 2D matrix of zeros of optimal size for a given points cloud
        """
        if not points_list:
            points_list = self.dataset[:][:, :-1]
            
        xmax, ymax = points_list.max(axis=0)
        xmin, ymin = points_list.min(axis=0)
        x_range = xmax - xmin
        y_range = ymax - ymin
        ranges = np.array([x_range, y_range])
        min_pixel_distance = self.prepare_min_distance(ranges, scale_factor)
        res_cols = int(math.ceil(ranges[0]/min_pixel_distance))
        res_rows = int(math.ceil(ranges[1]/min_pixel_distance))
        raster = np.zeros((res_rows, res_cols))
        return raster, xmin, xmax, ymin, ymax, min_pixel_distance
    
    def interpolate_raster(self, scale_factor=0.01, kriging_type='ordinary', number_of_neighbours=4,
                           update_model=False):
        """
        Interpolate raster of the size calculated from the distance between the top-left and bottom-down points
        from the points of known values. Pixel size is a smaller dimension from the two maximum distance dimensions
        multiplied by the scale factor (scale factor must be a fraction).
        :param scale_factor: Scale factor is multiplied by a smaller distance from width or height of
        a raster. This gives one unit (pixel width / height) of a created map. Scale factor must be a fraction.
        :param kriging_type: 'ordinary' or 'simple' Kriging
        :param number_of_neighbours: how many neighbouring points are spatially correlated with the unknown point
        :param update_model: If True then class object stores interpolated matrix and error matrix
        :return: [numpy 2D array (matrix) of interpolated values,
        numpy 2D array (matrix) of the estimated variance error]
        """
        raster_data = self.create_raster(scale_factor)
        raster = raster_data[0]
        error_mtx = np.zeros(raster.shape)
        raster_shape = np.shape(raster)
        for i in range(0, raster_shape[0]):
            dy = raster_data[4] - raster_data[5] * i
            for j in range(0, raster_shape[1]):
                dx = raster_data[1] + raster_data[5] * j
                self.prepare_data([dx, dy], number_of_records=number_of_neighbours)
                if kriging_type == 'simple':
                    s = self.simple_kriging()
                elif kriging_type == 'ordinary':
                    s = self.ordinary_kriging()
                raster[i, j] = s[0]
                error_mtx[i, j] = s[1]
        if update_model:
            self.interpolated_raster = raster
            self.interpolated_error_matrix = error_mtx
        else:
            return raster, error_mtx

    ##### VISUALIZATION METHODS #####

    def show_known_points(self):
        """
        Function shows known points distribution.
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=0.01)
        plt.title('Distribution of points with known values')
        plt.show()

    def show_results(self, values_matrix=None):
        """
        Function shows interpolated results.
        :param values_matrix: if values matrix is not given then it is assumed that object instance has this
        matrix stored
        """
        plt.figure(figsize=(10, 10))
        if values_matrix is None:
            plt.imshow(self.interpolated_raster, cmap="magma")
        else:
            plt.imshow(values_matrix, cmap='magma')
        plt.title('Interpolated values')
        plt.colorbar()
        plt.show()

    def show_error_matrix(self, error_matrix=None):
        """
        Function shows interpolated results.
        :param error_matrix: if error matrix is not given then it is assumed that object instance has this
        matrix stored
        """
        plt.figure(figsize=(10, 10))
        if error_matrix is None:
            plt.imshow(self.interpolated_error_matrix, cmap="magma")
        else:
            plt.imshow(error_matrix, cmap='magma')
        plt.title('Error matrix')
        plt.colorbar()
        plt.show()
