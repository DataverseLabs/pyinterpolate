import numpy as np
import matplotlib.pyplot as plt


class TheoreticalSemivariogram:
    """
    Class for calculating theoretical semivariogram. Class takes two parameters during initialization:
    points_array - analysed points where the last column is representing values, typically DEM
    empirical_semivariance - semivariance where first row of array represents lags and the second row
    represents semivariance's values for given lag

    Available methods:
    * predict(distances) - method predicts value of the unknown point based on the chosen model
    * fit_semivariance(model_type, number_of_ranges=200) - returns given model
    * find_optimal_model(weight_function, number_of_ranges=200) - returns optimal model for given 
    experimental semivariogram

    Static methods:
    * spherical_model(distance, nugget, sill, semivar_range)
    * gaussian_model(distance, nugget, sill, semivar_range)
    * exponential_model(distance, nugget, sill, semivar_range)
    * linear_model(distance, nugget, sill, semivar_range)
    
    Additional methods:
    * calculate_base_error() - calculates mean squared difference between exmperimental semivariogram
    and "flat line" of zero values
    
    Data visualization methods:
    * show_experimental_semivariogram() - shows semivariogram which is a part of the class object's 
    instance
    * show_semivariogram() - shows experimental semivariogram with theoretical model (if it was calculated)
    """

    def __init__(self, points_array, empirical_semivariance):
        self.points_values = points_array[:, -1]
        self.empirical_semivariance = empirical_semivariance
        self.theoretical_model = None
        self.params = None

    @staticmethod
    def spherical_model(distance, nugget, sill, semivar_range):
        """
        :param distance: array of ranges from empirical semivariance
        :param nugget: scalar
        :param sill: scalar
        :param semivar_range: optimal range calculated by fit_semivariance method
        :return x: an array of modeled values for given range. Values are calculated based on the spherical model.
        """

        x = np.where((distance <= semivar_range),
                     (nugget + sill * (3.0 * distance / (2.0 * semivar_range)) - (
                             (distance / semivar_range) ** 3.0 / 2.0)),
                     (nugget + sill))
        return x

    @staticmethod
    def gaussian_model(distance, nugget, sill, semivar_range):
        """
        :param distance: array of ranges from empirical semivariance
        :param nugget: scalar
        :param sill: scalar
        :param semivar_range: optimal range calculated by fit_semivariance method
        :return x: an array of modeled values for given range. Values are calculated based on the gaussian model.
        """
        x = nugget + sill * (1 - np.exp(-distance * distance / (semivar_range ** 2)))
        return x

    @staticmethod
    def exponential_model(distance, nugget, sill, semivar_range):
        """
        :param distance: array of ranges from empirical semivariance
        :param nugget: scalar
        :param sill: scalar
        :param semivar_range: optimal range calculated by fit_semivariance method
        :return x: an array of modeled values for given range. Values are calculated based on the exponential model.
        """
        x = nugget + sill * (1 - np.exp(-distance / semivar_range))
        return x

    @staticmethod
    def linear_model(distance, nugget, sill, semivar_range):
        """
        :param distance: array of ranges from empirical semivariance
        :param nugget: scalar
        :param sill: scalar
        :param semivar_range: optimal range calculated by fit_semivariance method
        :return x: an array of modeled values for given range. Values are calculated based on the linear model.
        """

        x = np.where((distance <= semivar_range),
                     (nugget + sill * (distance / semivar_range)),
                     (nugget + sill))
        return x

    def fit_semivariance(self, model_type, number_of_ranges=200):
        """
        :param model_type: 'exponential', 'gaussian', 'linear', 'spherical'
        :param number_of_ranges: deafult = 200. Used to create an array of equidistant ranges 
        between minimal range of empirical semivariance and maximum range of empirical semivariance.
        """

        # model
        models = {
            'spherical': self.spherical_model,
            'gaussian': self.gaussian_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
        }
        model = models[model_type]

        # sill
        sill = np.var(self.points_values)

        # nugget
        if self.empirical_semivariance[:, 0][0] != 0:
            nugget = self.empirical_semivariance[:, 0][0]
        else:
            nugget = 0

        # range
        minrange = self.empirical_semivariance[:, 0][1]
        maxrange = self.empirical_semivariance[:, 0][-1]
        ranges = np.linspace(minrange, maxrange, number_of_ranges)
        optimal_range = self.calculate_range(model, ranges, nugget, sill)

        # output model
        self.theoretical_model = model
        self.params = [nugget, sill, optimal_range]
        
    def find_optimal_model(self, weighted=False, number_of_ranges=200):
        """
        :param weighted: default=False. If True then each lag is weighted by:
                                        sqrt(N(h))/gamma_{exp}(h)
                                        where:
                                        N(h) - number of point pairs in a given range,
                                        gamma_{exp}(h) - value of experimental semivariogram for
                                        a given range.
        :param number_of_ranges: deafult = 200. Used to create an array of equidistant ranges 
        between minimal range of empirical semivariance and maximum range of empirical semivariance.
        """
        
        # models
        models = {
            'spherical': self.spherical_model,
            'gaussian': self.gaussian_model,
            'exponential': self.exponential_model,
            'linear': self.linear_model,
        }
        
        # calculate base error for a flat line
        base_error = self.calculate_base_error()

        # sill
        sill = np.var(self.points_values)

        # nugget
        if self.empirical_semivariance[:, 0][0] != 0:
            nugget = self.empirical_semivariance[:, 0][0]
        else:
            nugget = 0

        # range
        minrange = self.empirical_semivariance[:, 0][1]
        maxrange = self.empirical_semivariance[:, 0][-1]
        ranges = np.linspace(minrange, maxrange, number_of_ranges)
        
        # search for the best model
        model_name = 'null'
        for model in models:
            optimal_range = self.calculate_range(models[model], ranges, nugget, sill)
            
            # output model
            params = [nugget, sill, optimal_range]
            if not weighted:
                model_error = self.calculate_model_error(models[model], params)
            else:
                model_error = self.calculate_model_error(models[model], params, True)

            print('Model: {}, error value: {}'.format(model, model_error))

            if model_error < base_error:
                base_error = model_error
                self.theoretical_model = models[model]
                self.params = params
                model_name = model
                
        # print output
        print('########## Chosen model: {} ##########'.format(model_name))
        
    def calculate_range(self, model, ranges, nugget, sill):
        errors = []
        for r in ranges:
            x = (self.empirical_semivariance[:, 1] - model(self.empirical_semivariance[:, 0], nugget, sill, r))
            x = x ** 2
            errors.append(np.mean(x))
        optimal_rg = ranges[np.argmin(errors)]
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
        error = np.mean((self.empirical_semivariance[:, 1] - zeros)**2)
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
            error = nh/vals * np.abs(vals - model(self.empirical_semivariance[:, 0], par[0], par[1], par[2]))

        error = np.mean(error)
        return error

    def predict(self, distances):
        """
        :param distances: array of distances from points of known locations and values to the point of 
        unknown value
        :return: model with predicted values
        """
        
        output_model = self.theoretical_model(distances,
                                              self.params[0],
                                              self.params[1],
                                              self.params[2])
        return output_model

    def show_experimental_semivariogram(self):
        """
        Function shows experimental semivariogram of a given model
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.empirical_semivariance[:, 0], self.empirical_semivariance[:, 1], color='blue')
        plt.title('Experimental semivariogram')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    def show_semivariogram(self):
        """
        Function shows experimental and theoretical semivariogram in one plot
        """
        if self.theoretical_model is None:
            raise AttributeError('Theoretical semivariogram is not calculated. \
            Did you run fit_semivariance(model_type, number_of_ranges) on your model?')
        else:
            x = self.calculate_values()
            plt.figure(figsize=(12, 12))
            plt.plot(self.empirical_semivariance[:, 0], self.empirical_semivariance[:, 1], color='blue')
            plt.plot(self.empirical_semivariance[:, 0], x, color='red')
            plt.legend(['Empirical semivariogram', 'Theoretical semivariogram'])
            plt.title('Empirical and theoretical semivariogram comparison')
            plt.xlabel('Distance')
            plt.ylabel('Semivariance')
            plt.show()
