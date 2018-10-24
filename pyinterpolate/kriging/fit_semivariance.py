import numpy as np
import matplotlib.pyplot as plt


class TheoreticalSemivariogram:
    """
    Class for calculating theoretical semivariogram. Class takes two parameters during initialization:
    points_array - analysed points where the last column is representing values, typically DEM
    empirical_semivariance - semivariance where first row of array represents lags and the second row
    represents semivariance's values for given lag

    Available methods:
    * fit_semivariance(model_type, number_of_ranges=200)

    Static methods:
    * spherical_model(distance, nugget, sill, semivar_range)
    * gaussian_model(distance, nugget, sill, semivar_range)
    * exponential_model(distance, nugget, sill, semivar_range)
    * linear_model(distance, nugget, sill, semivar_range)
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
        :return x: an array of modelled values for given range. Values are calculated based on the spherical model.
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
        :return x: an array of modelled values for given range. Values are calculated based on the gaussian model.
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
        :return x: an array of modelled values for given range. Values are calculated based on the exponential model.
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
        :return x: an array of modelled values for given range. Values are calculated based on the linear model.
        """

        x = np.where((distance <= semivar_range),
                     (nugget + sill * (distance / semivar_range)),
                     (nugget + sill))
        return x

    def calculate_range(self, model, ranges, nugget, sill):
        errors = []
        for r in ranges:
            x = (self.empirical_semivariance[1] - model(self.empirical_semivariance[0], nugget, sill, r))
            x = x ** 2
            errors.append(np.mean(x))
        optimal_rg = ranges[np.argmin(errors)]
        return optimal_rg

    def fit_semivariance(self, model_type, number_of_ranges=200):
        """
        :param model_type: 'exponential', 'gaussian', 'linear', 'spherical'
        :param number_of_ranges: deafult = 200. Used to create an array of equidistant ranges between minimal range of
        empirical semivariance and maximum range of empirical semivariance.
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

        # nugget / check if this is true
        if self.empirical_semivariance[0][0] != 0:
            nugget = 0.0
        else:
            nugget = self.empirical_semivariance[0][0]

        # range
        minrange = self.empirical_semivariance[0][1]
        maxrange = self.empirical_semivariance[0][-1]
        ranges = np.linspace(minrange, maxrange, number_of_ranges)
        optimal_range = self.calculate_range(model, ranges, nugget, sill)

        # output model
        self.theoretical_model = model
        self.params = [nugget, sill, optimal_range]

    def calculate_values(self):
        output_model = self.theoretical_model(self.empirical_semivariance[0],
                                              self.params[0],
                                              self.params[1],
                                              self.params[2])
        return output_model

    def predict(self, distances):
        """
        :param distances: array of distances from points of known locations and values to the point of unknown value
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
        plt.plot(self.empirical_semivariance[0], self.empirical_semivariance[1], color='blue')
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
            plt.plot(self.empirical_semivariance[0], self.empirical_semivariance[1], color='blue')
            plt.plot(self.empirical_semivariance[0], x, color='red')
            plt.legend(['Empirical semivariogram', 'Theoretical semivariogram'])
            plt.title('Empirical and theoretical semivariogram comparison')
            plt.xlabel('Distance')
            plt.ylabel('Semivariance')
            plt.show()