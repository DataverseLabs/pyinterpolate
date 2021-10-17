"""
Authors:

Scott Gallacher | @scottgallacher-3
Szymon Molinski | @szymon-datalions

Contributors:

Ethem Turgut | https://github.com/ethmtrgt

"""


import csv
from operator import itemgetter
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class TheoreticalSemivariogram:
    """
    Class calculates theoretical semivariogram. Class takes two parameters during initialization:

    points_array - (numpy array) analysed points where the last column represents values, typically x, y, value,
    empirical_semivariance - (numpy array) semivariance where first row of array represents lags and the second row
        represents semivariance's values for a given lag.

    Available methods:

    - predict() - method predicts value of the unknown point based on the chosen model,
    - fit_semivariance() - Method fits experimental points into chosen semivariance model type,
    - find_optimal_model() - Method fits experimental points into all available models and choose one with the lowest
        error.

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

        self.points_array = points_array
        self.empirical_semivariance = empirical_semivariance
        self.verbose = verbose
        self.theoretical_model = None
        self.chosen_model_name = None
        self.nugget = None
        self.range = None
        self.sill = None
        self.model_error = None
        self.is_weighted = False

    # MODELS

    @staticmethod
    def spherical_model(lags, nugget, sill, semivar_range):
        """

        gamma = nugget + sill*[(3/2)*a - (1/2)*(a**3)], 0 <= lag <= range
        gamma = nugget + sill, lag > range
        gamma = 0, lag == 0

        where:

        a = lag / range

        INPUT:

        :param lags: array of lags from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the spherical model.
        """

        a = lags / semivar_range
        a1 = 3 / 2 * a
        a2 = 1 / 2 * a ** 3

        gamma = np.where((lags <= semivar_range),
                         (nugget + sill * (a1 - a2)),
                         (nugget + sill))

        return gamma

    @staticmethod
    def exponential_model(lags, nugget, sill, semivar_range):
        """

        gamma = nugget + sill*[1 - exp(-lag/range)], distance > 0
        gamma = 0, lag == 0

        INPUT:

        :param lags: array of lags from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the exponential model.
        """

        try:
            gamma = nugget + sill * (1 - np.exp(-lags / semivar_range))
        except TypeError:
            lags = lags.astype(float)
            semivar_range = float(semivar_range)
            gamma = nugget + sill * (1 - np.exp(-lags / semivar_range))

        return gamma

    @staticmethod
    def linear_model(lags, nugget, sill, semivar_range):
        """

        gamma = nugget + sill*(lag/range), 0 <= lag <= range
        gamma = nugget + sill, lag > range
        gamma = 0, lag == 0


        INPUT:

        :param lags: array of lags from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the linear model.
        """

        gamma = np.where((lags <= semivar_range),
                         (nugget + sill * (lags / semivar_range)),
                         (nugget + sill))

        return gamma

    @staticmethod
    def gaussian_model(lags, nugget, sill, semivar_range):
        """

        gamma = nugget + sill*[1 - exp(-1*(lag**2 / range**2))], lag > 0
        gamma = 0, lag == 0

        INPUT:

        :param lags: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the gaussian model.
        """
        gamma = nugget + sill * (1 - np.exp(-1*(lags ** 2 / semivar_range ** 2)))

        if lags[0] == 0:
            gamma[0] = 0

        return gamma
    
    @staticmethod
    def power_model(lags, nugget, sill, semivar_range):
        """

        gamma = nugget + sill*[1 - exp(lag**2 / range**2)], lag > 0
        gamma = 0, lag == 0

        INPUT:

        :param lags: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the power model.
        """
        
        gamma = nugget + sill * (1 - np.exp((lags ** 2 / semivar_range ** 2)))

        if lags[0] == 0:
            gamma[0] = 0

        return gamma
    
    @staticmethod
    def cubic_model(lags, nugget, sill, semivar_range):
        """

        gamma = nugget + sill*[7*(a**2) - 8.75*(a**3) + 3.5*(a**5) - 0.75*(a**7)], lag < range
        gamma = nugget + sill, lag > range
        gamma = 0, lag == 0

        where:

        a = lag / range

        INPUT:

        :param lags: array of lags from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the cubic model.
        """

        a = lags / semivar_range
        a1 = 7 * a ** 2
        a2 = -8.75 * a ** 3
        a3 = 3.5 * a ** 5
        a4 = -0.75 * a ** 7
        
        gamma = np.where((lags < semivar_range), nugget + sill * (a1 + a2 + a3 + a4), nugget + sill)
        
        if lags[0] == 0:
            gamma[0] = 0

        return gamma
    
    @staticmethod
    def circular_model(lags, nugget, sill, semivar_range):
        ##### NOTE: found two competing model formulae for the circular model
        ##### 1st one doesn't seem to work with the test data; but 2nd one does
        ##### Sources added in docstring, further comparison may be needed
        ##### (DELETE AFTER REVIEW)
        """

        gamma = nugget + sill*[1 - (2/np.pi * np.arccos(a)) + np.sqrt(1 - (lag ** 2)/ (range ** 2) )], 0 < lag <= range
        OR gamma = nugget + (2/np.pi)*sill*[a * np.sqrt(1 - a ** 2) + np.arcsin(a)], 0 < lag <= range
        gamma = 0, lag == 0
        
        where:
        
        a = lag / range
        
        (Model 1 Source:
        https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-kriging-works.htm#GUID-94A34A70-DBCF-4B23-A198-BB50FB955DC0)
        (Model 2 Source:
        https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-kriging-works.htm#GUID-94A34A70-DBCF-4B23-A198-BB50FB955DC0)

        INPUT:

        :param lags: array of ranges from empirical semivariance,
        :param nugget: scalar,
        :param sill: scalar,
        :param semivar_range: optimal range calculated by fit_semivariance method.

        OUTPUT:

        :return: an array of modeled values for given range. Values are calculated based on the circular model.
        """
        # TODO: check conditions:
        # apparently, even using np.where uncovers invalid values in the arccos and square root
        # but as long as lag <= range this shouldn't happen
        # use np.clip on the arrays to be passed
        a = lags / semivar_range
        
        # use np.clip to limit range of values passed into np.arccos and np.sqrt
        # gamma = np.where((lags <= semivar_range),
        #                  (nugget + sill*(1 - 2/np.pi * np.arccos(np.clip(a, -1, 1)) *
        #                      np.sqrt(1 - np.clip(a**2, -1, 1))) ),
        #                  (nugget + sill))
        
        # second formula found which seems to fit better, and looks as expected
        gamma = nugget + (2/np.pi) * sill*(a * np.sqrt(1 - np.clip(a**2, -1, 1)) + np.arcsin(np.clip(a, -1, 1)))

        if lags[0] == 0:
            gamma[0] = 0

        return gamma

    def fit_semivariance(self, model_type, number_of_ranges=16):
        """
        Method fits experimental points into chosen semivariance model type.

        INPUT:

        :param model_type: (str) 'exponential', 'gaussian', 'linear', 'spherical',
        :param number_of_ranges: (int) deafult = 16. Used to create an array of equidistant ranges
            between minimal range of empirical semivariance and maximum range of empirical semivariance.

        OUTPUT:

        :return: (model_type, model parameters)
        """

        # model
        models = {
            'spherical': self.spherical_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
            'gaussian': self.gaussian_model
        }
        model = models[model_type]
        self.chosen_model_name = model_type
        self.theoretical_model = model

        # sill
        sill = np.var(self.points_array[:, -1])

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
        self.nugget = nugget
        self.sill = sill
        self.range = optimal_range

        # model error
        self.model_error = self.calculate_model_error(model, self.nugget, self.sill, self.range)

        return model_type

    def find_optimal_model(self, weighted=False, number_of_ranges=16):
        """

        Method fits experimental points into all available models and choose one with the lowest error.

        INPUT:

        :param weighted: (bool) default=False. If True then each lag is weighted by:

            sqrt(N(h))/gamma_{exp}(h)

            where:

            - N(h) - number of point pairs in a given range, gamma_{exp}(h) - value of experimental semivariogram for h.

        :param number_of_ranges: (int) default=16. Used to create an array of equidistant ranges
            between minimal range of empirical semivariance and maximum range of empirical semivariance.
        """

        if weighted:
            self.is_weighted = True

        # models
        models = {
            'spherical': self.spherical_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
        }

        # calculate base error for a flat line
        base_error = self.calculate_base_error()

        # sill
        sill = np.var(self.points_array[:, -1])

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
        model_errors = [('Linear (LS) reference model', base_error, None)]
        for model in models:
            optimal_range = self.calculate_range(models[model], ranges, nugget, sill)

            # output model
            model_error = self.calculate_model_error(models[model], nugget, sill, optimal_range)

            model_errors.append((model, model_error, [nugget, sill, optimal_range]))
            if self.verbose:
                print('Model: {}, error value: {}'.format(model, model_error))

        # Select the best model
        sorted_errors = sorted(model_errors, key=itemgetter(1))

        if sorted_errors[0][0] == 'Linear (LS) reference model':
            # This is unlikely case when error estimated as the squared distance between extrapolated values and
            # x axis is smaller than models' errors

            model_name = sorted_errors[1][0]
            model_error = sorted_errors[1][1]
            model_params = sorted_errors[1][2]

            warning_msg = 'WARNING: linear model fitted to the experimental variogram is better than the core models!'
            warnings.warn(warning_msg)
            if self.verbose:
                print('Chosen model: {}, with value of: {}.'.format(
                    model_name, model_error
                ))
        else:
            model_name = sorted_errors[0][0]
            model_error = sorted_errors[0][1]
            model_params = sorted_errors[0][2]
            if self.verbose:
                print('Chosen model: {}, with value: {}.'.format(
                    model_name, model_error
                ))

        self.theoretical_model = models[model_name]
        self.nugget = model_params[0]
        self.sill = model_params[1]
        self.range = model_params[2]
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
                                              self.nugget,
                                              self.sill,
                                              self.range)
        return output_model

    @staticmethod
    def _curve_fit_function(x, a, b):
        """
        Method fits data into a 1st order polynomial curve where:
            y = a * x + b

        INPUT:

        :param a: number or numpy array,
        :param b: number or numpy array,
        :param x: number or numpy array.

        OUTPUT:

        :return: y -> a * x + b | number or numpy array.
        """

        y = a * x + b
        return y

    def _get_weights(self):
        """
        Method creates weights based on the lags for each semivariogram point

        OUTPUT:

        :returns: (numpy array)
        """

        nh = np.sqrt(self.empirical_semivariance[:, 2])
        vals = self.empirical_semivariance[:, 1]
        nh_divided_by_vals = np.divide(nh,
                                       vals,
                                       out=np.zeros_like(nh),
                                       where=vals != 0)
        return nh_divided_by_vals

    def calculate_base_error(self):
        """
        Method calculates base error as the difference between the least squared model
            of experimental semivariance and the experimental semivariance points.

        OUTPUT:

        :returns: (float) mean squared difference error
        """

        popt, _pcov = curve_fit(self._curve_fit_function,
                                self.empirical_semivariance[:, 0],
                                self.empirical_semivariance[:, 1])
        a, b = popt
        y = self._curve_fit_function(self.empirical_semivariance[:, 0],
                                     a, b)
        error = np.sqrt((self.empirical_semivariance[:, 1] - y) ** 2)

        if not self.is_weighted:
            mean_error = np.mean(error)
            return mean_error
        else:
            weights = self._get_weights()
            mean_error = np.mean(weights * error)
            return mean_error

    def calculate_model_error(self, model, nugget, sill, semivar_range):
        """
        Function calculates error between specific models and experimental curve.

        OUTPUT:

        :returns: (float) mean squared difference between model and experimental variogram.
        """
        error = np.sqrt((self.empirical_semivariance[:, 1] - model(self.empirical_semivariance[:, 0],
                                                                   nugget,
                                                                   sill,
                                                                   semivar_range)) ** 2)
        if not self.is_weighted:
            return np.mean(error)
        else:
            weights = self._get_weights()
            return np.mean(weights * error)

    def predict(self, distances):
        """

        INPUT:

        :param distances: array of distances from points of known locations and values to the point of
        unknown value,

        OUTPUT:
        :return: model with predicted values.
        """

        output_model = self.theoretical_model(distances,
                                              self.nugget,
                                              self.sill,
                                              self.range)
        return output_model

    def export_model(self, filename):
        """
        Function exports semivariance model to the csv file with columns:

        - name: [model name],
        - nugget: [value],
        - sill: [value],
        - range: [value],
        - model_error: [value]"""

        model_parameters = {
            'name': self.chosen_model_name,
            'nugget': self.nugget,
            'sill': self.sill,
            'range': self.range,
            'model_error': self.model_error
        }

        csv_cols = list(model_parameters.keys())
        try:
            with open(filename, 'w') as semivar_csv:
                writer = csv.DictWriter(semivar_csv, fieldnames=csv_cols)
                writer.writeheader()
                writer.writerow(model_parameters)
        except IOError:
            raise IOError("I/O error, provided path for semivariance parameters is not valid")

    def import_model(self, filename):
        """

        Function imports semivariance model and updates it's parameters
        (model name, nugget, sill, range, model error)."""

        models = {
            'spherical': self.spherical_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
            'gaussian': self.gaussian_model,
        }

        csv_cols = ['name', 'nugget', 'sill', 'range', 'model_error']
        try:
            with open(filename, 'r') as semivar_csv:
                reader = csv.DictReader(semivar_csv, fieldnames=csv_cols)
                next(reader)
                for row in reader:
                    self.nugget = float(row['nugget'])
                    self.sill = float(row['sill'])
                    self.range = float(row['range'])
                    self.chosen_model_name = row['name']
                    if row['model_error']:
                        self.model_error = float(row['model_error'])
                    else:
                        self.model_error = None
                    try:
                        self.theoretical_model = models[self.chosen_model_name]
                    except KeyError:
                        raise KeyError('You have provided wrong model name. Available names: spherical, gaussian, '
                                       'exponential, linear.')
        except IOError:
            raise IOError("I/O error, provided path for semivariance parameters is not valid")

    def export_semivariance(self, filename):
        """
        Function exports empirical and theoretical semivariance models into csv file.

        INPUT:

        :param filename: (str) Path to the csv file to be stored.
        """

        if self.theoretical_model is None:
            raise RuntimeError('Theoretical semivariogram is not calculated. \
            Did you run fit_semivariance(model_type, number_of_ranges) on your model?')

        if not isinstance(filename, str):
            raise ValueError('Given path is not a string type')

        if not filename.endswith('.csv'):
            filename = filename + '.csv'

        # Create DataFrame to export

        theo_values = self.calculate_values()
        dt = {
            'lag': self.empirical_semivariance[:, 0],
            'experimental': self.empirical_semivariance[:, 1],
            'theoretical': theo_values
        }
        df = pd.DataFrame.from_dict(dt, orient='columns')
        df.to_csv(filename, index=False)

    def show_experimental_semivariogram(self):
        """
        Function shows experimental semivariogram of a given model.
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.empirical_semivariance[:, 0], self.empirical_semivariance[:, 1], 'bo')
        plt.title('Experimental semivariogram')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    def show_semivariogram(self):
        """
        Function shows experimental and theoretical semivariogram in one plot.
        """
        if self.theoretical_model is None:
            raise RuntimeError('Theoretical semivariogram is not calculated. \
            Did you run fit_semivariance(model_type, number_of_ranges) on your model?')

        x = self.calculate_values()
        plt.figure(figsize=(12, 12))
        plt.plot(self.empirical_semivariance[:, 0], self.empirical_semivariance[:, 1], 'bo')
        plt.plot(self.empirical_semivariance[:, 0], x, color='red')
        plt.legend(['Empirical semivariogram', 'Theoretical semivariogram - {} model'.format(
            self.chosen_model_name
        )])
        title_text = 'Empirical and theoretical semivariogram comparison, model error = {:.2f}'.format(
            self.model_error
        )
        plt.title(title_text)
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()
