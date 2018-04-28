import numpy as np


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
        :return: Theoretical model of semivariance (values only)
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
        output_model = model(self.empirical_semivariance[0], nugget, sill, optimal_range)
        return output_model
