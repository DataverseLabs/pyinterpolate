import csv
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt


class TheoreticalSemivariogram:
    """
    Class for calculating theoretical semivariogram. Class takes two parameters during initialization:
    points_array - analysed points where the last column is representing values, typically DEM
    empirical_semivariance - semivariance where first row of array represents lags and the second row
    represents semivariance's values for given lag

    Available methods:

    - predict() - method predicts value of the unknown point based on the chosen model,
    - fit_semivariance() - returns given model,
    - find_optimal_model() - returns optimal model for given experimental semivariogram.

    Available theoretical models:

    - spherical_model(distance, nugget, sill, semivar_range)
    - gaussian_model(distance, nugget, sill, semivar_range)
    - exponential_model(distance, nugget, sill, semivar_range)
    - linear_model(distance, nugget, sill, semivar_range)

    Additional methods:

    - calculate_base_error(),
    - show_experimental_semivariogram() - shows semivariogram which is a part of the class object's instance,
    - show_semivariogram() - shows experimental semivariogram with theoretical model (if it was calculated).
    """

    def __init__(self, points_array=None, empirical_semivariance=None, verbose=False):
        """
        INPUT:

        :param points_array: (numpy array) [point x, point y, value] (optional if model parameters are imported)
        :param empirical_semivariance: (numpy array) array of pair of lag and semivariance values where
                 semivariance[:, 0] = array of lags
                 semivariance[:, 1] = array of lag's values
                 semivariance[:, 2] = array of number of points in each lag.
                 (optional if model parameters are imported)
        :param verbose: (bool) if True then all messages are printed, otherwise nothing.
        """

        self.points_values = points_array
        self.empirical_semivariance = empirical_semivariance
        self.verbose = verbose
        self.theoretical_model = None
        self.chosen_model_name = None
        self.params = None
        self.model_error = None

    # MODELS

    @staticmethod
    def spherical_model(distance, nugget, sill, semivar_range):
        """

        INPUT:

        :param distance: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the spherical model.
        """

        x = np.where((distance <= semivar_range),
                     (nugget + sill * (3.0 * distance / (2.0 * semivar_range)) - (
                             (distance / semivar_range) ** 3.0 / 2.0)),
                     (nugget + sill))
        return x

    @staticmethod
    def exponential_model(distance, nugget, sill, semivar_range):
        """

        INPUT:

        :param distance: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the exponential model.
        """
        try:
            x = nugget + sill * (1 - np.exp(-distance / semivar_range))
        except TypeError:
            distance = distance.astype(float)
            semivar_range = float(semivar_range)
            x = nugget + sill * (1 - np.exp(-distance / semivar_range))
        return x

    @staticmethod
    def linear_model(distance, nugget, sill, semivar_range):
        """

        INPUT:

        :param distance: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the linear model.
        """

        x = np.where((distance <= semivar_range),
                     (nugget + sill * (distance / semivar_range)),
                     (nugget + sill))
        return x

    @staticmethod
    def gaussian_model(distance, nugget, sill, semivar_range):
        """

        INPUT:

        :param distance: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the gaussian model.
        """
        x = nugget + sill * (1 - np.exp(-distance * distance / (semivar_range ** 2)))
        return x

    def fit_semivariance(self, model_type, number_of_ranges=16):
        """

        INPUT:

        :param model_type: 'exponential', 'gaussian', 'linear', 'spherical',
        :param number_of_ranges: deafult = 16. Used to create an array of equidistant ranges
            between minimal range of empirical semivariance and maximum range of empirical semivariance.

        OUTPUT:

        :return: (model_type, model parameters)
        """

        # model
        models = {
            'spherical': self.spherical_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
            'gaussian': self.gaussian_model,
        }
        model = models[model_type]
        self.chosen_model_name = model_type
        self.theoretical_model = model

        # sill
        sill = np.var(self.points_values[:, -1])

        # nugget
        if self.empirical_semivariance[0][0] != 0:
            nugget = 0
        else:
            nugget = self.empirical_semivariance[0][1]

        # range
        minrange = self.empirical_semivariance[:, 0][1]
        maxrange = self.empirical_semivariance[:, 0][-1]
        ranges = np.linspace(minrange, maxrange, number_of_ranges)
        optimal_range = self.calculate_range(model, ranges, nugget, sill)

        # output model
        self.params = [nugget, sill, optimal_range]

        # model error
        self.model_error = self.calculate_model_error(model, self.params)

        return (model_type, self.params)

    def find_optimal_model(self, weighted=False, number_of_ranges=16):
        """

        INPUT:

        :param weighted: default=False. If True then each lag is weighted by:

            sqrt(N(h))/gamma_{exp}(h)

            where:

            - N(h) - number of point pairs in a given range, gamma_{exp}(h) - value of experimental semivariogram for h.

        :param number_of_ranges: deafult = 16. Used to create an array of equidistant ranges
            between minimal range of empirical semivariance and maximum range of empirical semivariance.
        """

        # models
        models = {
            'spherical': self.spherical_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
        }

        # calculate base error for a flat line
        base_error = self.calculate_base_error()

        # sill
        sill = np.var(self.points_values[:, -1])

        # nugget
        if self.empirical_semivariance[0][0] != 0:
            nugget = 0
        else:
            nugget = self.empirical_semivariance[0][1]

        # range
        minrange = self.empirical_semivariance[:, 0][1]
        maxrange = self.empirical_semivariance[:, 0][-1]
        ranges = np.linspace(minrange, maxrange, number_of_ranges)

        # Calculate model errors
        model_errors = [('NULL model', base_error, None)]
        for model in models:
            optimal_range = self.calculate_range(models[model], ranges, nugget, sill)

            # output model
            params = [nugget, sill, optimal_range]
            if not weighted:
                model_error = self.calculate_model_error(models[model], params)
            else:
                model_error = self.calculate_model_error(models[model], params, True)

            model_errors.append((model, model_error, params))
            if self.verbose:
                print('Model: {}, error value: {}'.format(model, model_error))

        # Select the best model
        sorted_errors = sorted(model_errors, key=itemgetter(1))

        if sorted_errors[0][0] == 'NULL model':
            # This is unlikely case when error estimated as the squared distance between extrapolated values and
            # x axis is smaller than models' errors

            model_name = sorted_errors[1][0]
            model_error = sorted_errors[1][1]
            model_params = sorted_errors[1][2]
            self.theoretical_model = models[model_name]
            self.params = model_params

            print('WARNING: NULL model error is better than estimated models, its value is:', sorted_errors[0][1])
            print('Chosen model: {}, with value of: {}.'.format(
                model_name, model_error
            ))

        else:

            model_name = sorted_errors[0][0]
            model_error = sorted_errors[0][1]
            model_params = sorted_errors[0][2]
            self.theoretical_model = models[model_name]
            self.params = model_params
            if self.verbose:
                print('Chosen model: {}, with value: {}.'.format(
                    model_name, model_error
                ))
        self.chosen_model_name = model_name
        self.model_error = model_error
        return model_name

    def calculate_range(self, model, ranges, nugget, sill):
        errors = []
        for r in ranges:
            x = (self.empirical_semivariance[:, 1] - model(self.empirical_semivariance[:, 0], nugget, sill, r))
            x = x ** 2
            errors.append(np.mean(x))
        range_pos = np.argmin(errors)
        optimal_rg = ranges[range_pos]
        return optimal_rg

    def calculate_values(self):
        output_model = self.theoretical_model(self.empirical_semivariance[:, 0],
                                              self.params[0],
                                              self.params[1],
                                              self.params[2])
        return output_model

    def calculate_base_error(self):
        """
        Method calculates base error as a squared difference between experimental semivariogram and
        a "flat line" on the x-axis (only zeros)
        """
        n = len(self.empirical_semivariance[:, 1])
        zeros = np.zeros(n)
        error = np.mean(np.abs(self.empirical_semivariance[:, 1] - zeros))
        return error

    def calculate_model_error(self, model, par, weight=False):
        if not weight:
            error = np.abs(self.empirical_semivariance[:, 1] - model(self.empirical_semivariance[:, 0],
                                                                     par[0],
                                                                     par[1],
                                                                     par[2]))
        else:
            nh = np.sqrt(self.empirical_semivariance[:, 2])
            vals = self.empirical_semivariance[:, 1]
            nh_divided_by_vals = np.divide(nh,
                                           vals,
                                           out=np.zeros_like(nh),
                                           where=vals != 0)
            error = np.abs(nh_divided_by_vals *
                           (vals - model(self.empirical_semivariance[:, 0], par[0], par[1], par[2])))
        error = np.mean(error)
        return error

    def predict(self, distances):
        """

        INPUT:

        :param distances: array of distances from points of known locations and values to the point of
        unknown value,

        OUTPUT:
        :return: model with predicted values.
        """

        output_model = self.theoretical_model(distances,
                                              self.params[0],
                                              self.params[1],
                                              self.params[2])
        return output_model

    def export_model(self, filename):
        """Function exports semivariance model to the csv file with columns:

        - name: [model name],
        - nugget: [value],
        - sill: [value],
        - range: [value]"""

        model_parameters = {
            'name': self.chosen_model_name,
            'nugget': self.params[0],
            'sill': self.params[1],
            'range': self.params[2],
        }

        csv_cols = list(model_parameters.keys())
        try:
            with open(filename, 'w') as semivar_csv:
                writer = csv.DictWriter(semivar_csv, fieldnames=csv_cols)
                writer.writeheader()
                writer.writerow(model_parameters)
        except IOError:
            print("I/O error, provided path is not valid")

    def import_model(self, filename):
        """Function imports semivariance model and updates it's parameters"""

        models = {
            'spherical': self.spherical_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
            'gaussian': self.gaussian_model,
        }

        csv_cols = ['name', 'nugget', 'sill', 'range']
        try:
            with open(filename, 'r') as semivar_csv:
                reader = csv.DictReader(semivar_csv, fieldnames=csv_cols)
                next(reader)
                for row in reader:
                    self.params = [float(row['nugget']), float(row['sill']), float(row['range'])]
                    self.chosen_model_name = row['name']
                    try:
                        self.theoretical_model = models[self.chosen_model_name]
                    except KeyError:
                        raise KeyError('You have provided wrong model name. Available names: spherical, gaussian, '
                                       'exponential, linear.')
        except IOError:
            print("I/O error, provided path is not valid")

    def show_experimental_semivariogram(self):
        """
        Function shows experimental semivariogram of a given model.
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.empirical_semivariance[:, 0], self.empirical_semivariance[:, 1], color='blue')
        plt.title('Experimental semivariogram')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    def show_semivariogram(self):
        """
        Function shows experimental and theoretical semivariogram in one plot.
        """
        if self.theoretical_model is None:
            raise AttributeError('Theoretical semivariogram is not calculated. \
            Did you run fit_semivariance(model_type, number_of_ranges) on your model?')
        
        x = self.calculate_values()
        plt.figure(figsize=(12, 12))
        plt.plot(self.empirical_semivariance[:, 0], self.empirical_semivariance[:, 1], color='blue')
        plt.plot(self.empirical_semivariance[:, 0], x, color='red')
        plt.legend(['Empirical semivariogram', 'Theoretical semivariogram - {} model'.format(
            self.chosen_model_name
        )])
        title_text = 'Empirical and theoretical semivariogram comparison, model error ={:.2f}'.format(
            self.model_error
        )
        plt.title(title_text)
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()
