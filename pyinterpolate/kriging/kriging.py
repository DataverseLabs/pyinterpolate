import math
import numpy as np
import matplotlib.pyplot as plt
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance


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
        predicted = self.model.predict(distances.ravel())
        predicted = np.matrix(predicted.reshape(n, n))

        w = np.linalg.solve(predicted, k)
        r = self.prepared_data[:, -2] - area_mean
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
            self.interpolate_raster = raster
            self.interpolated_error_matrix = error_mtx
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