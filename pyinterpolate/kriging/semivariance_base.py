# numerical libraries
import numpy as np

# spatial libraries
import geopandas as gpd

# pyinterpolate scripts
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance
from pyinterpolate.kriging.helper_functions.get_centroids import get_centroids


def calculate_weighted_semivariance(areal_centroids, total_population_per_unit,
                                    lags, step_size):
    """Function calculates weighted semivariance following Monestiez et al.:

    A. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C: Comparison of model based
    geostatistical methods in ecology: application to fin whale spatial distribution in
    northwestern Mediterranean Sea. In Geostatistics Banff 2004 Volume 2.
    Edited by: Leuangthong O, Deutsch CV. Dordrecht, The Netherlands, Kluwer Academic Publishers;
    2005:777-786.

    B. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C: Geostatistical modelling
    of spatial distribution of Balenoptera physalus in the northwestern Mediterranean Sea
    from sparse count data and heterogeneous observation efforts. Ecological Modelling 2006 in press.

    Equation for calculation is:

    s(h) = [1 / (2 * SUM(a=1, N(h)) (n(u_a) * n(u_a + h)) /...
                                  / (n(u_a) + n(u_a + h))
                )
           ] * SUM(a=1, N(h)) {

           [(n(u_a) * n(u_a + h)) / (n(u_a) + n(u_a + h))] *...
           * [(z(u_a) - z(u_a + h))^2] - m'

           }

    where:

    s(h) - Semivariogram of the risk,
    n(u_a) - size of the population at risk in the unit a,
    z(u_a) - mortality rate at the unit a,
    u_a + h - area at the distance (h) from the analyzed area,
    m' - population weighted mean of rates.

    INPUT:
    :param areal_centroids: numpy array [px, py, rate, id],
    :param total_population_per_unit: numpy array [id, population]
    :param lags: array of lags [lag1, lag2, lag...]
    :param step_size: step size of search radius.


    OUTPUT:
    :return: semivariance: numpy array of pair of lag and semivariance values where
             semivariance[0] = array of lags
             semivariance[1] = array of lag's values
             semivariance[2] = array of number of points in each lag.
    """

    # Calculate distances between centroids

    distances_array = calculate_distance(areal_centroids[:, :2])

    # Prepare semivariance arrays

    smv = []
    semivariance = []

    # Calculate semivariances

    # Prepare distances

    for h in lags:
        gammas = []
        populations = []
        weighted_mean = []
        distances_in_range = np.where(
            np.logical_and(
                np.greater_equal(distances_array, h - step_size),
                np.less_equal(distances_array, h + step_size))
        )

        for i in range(0, len(distances_in_range[0])):
            row_x = distances_in_range[0][i]
            row_x_h = distances_in_range[1][i]

            # Values (rates)

            gp1 = areal_centroids[row_x][2]
            gp2 = areal_centroids[row_x_h][2]

            # Populations

            pop1 = total_population_per_unit[areal_centroids[row_x][-1]]
            pop2 = total_population_per_unit[areal_centroids[row_x_h][-1]]

            # Population weighting

            weighted_pop = (pop1 * pop2) / (pop1 + pop2)
            populations.append(weighted_pop)

            # Population weighted mean of rates

            pop_sum = pop1 + pop2
            weighted_mean.append(((pop1 * gp1) + (pop2 * gp2)) / pop_sum)

            # Rates calculation

            g = weighted_pop * ((gp1 - gp2) ** 2)
            gammas.append(g)

        if len(gammas) == 0:
            gamma = 0
        else:
            gam_sum = np.sum(gammas)
            pop_sum = 2 * np.sum(populations)
            weight_sum = np.sum(weighted_mean)
            gamma = (gam_sum - weight_sum) / pop_sum

        smv.append([gamma, len(gammas)])

    # Selecting semivariances
    for i in range(len(lags)):
        if smv[i][0] > 0:
            semivariance.append([lags[i], smv[i][0], smv[i][1]])
        else:
            semivariance.append([lags[i], 0, 0])

    semivariance = np.vstack(semivariance)
    return semivariance


def calculate_semivariance(data, lags, step_size):
    """Function calculates semivariance of a given set of points.
    
    INPUT:
    :param data: array of coordinates and their values,
    :param lags: array of lags between points,
    :param step_size: distance which should be included in the gamma parameter which enhances range of interest.
    
    OUTPUT:
    :return: semivariance: numpy array of pair of lag and semivariance values where
             semivariance[0] = array of lags
             semivariance[1] = array of lag's values
             semivariance[2] = array of number of points in each lag.
             
    WARNING:
    Function will be deprecated in the final version of the library. Its properties will be covered by the
    Semivariance class object."""
    
    distances_array = calculate_distance(data[:, :-1])
    smv = []
    semivariance = []

    # Calculate semivariances
    for h in lags:
        gammas = []
        distances_in_range = np.where(
            np.logical_and(
                np.greater_equal(distances_array, h - step_size), np.less_equal(distances_array, h + step_size)))
        for i in range(0, len(distances_in_range[0])):
            row_x = distances_in_range[0][i]
            row_x_h = distances_in_range[1][i]
            gp1 = data[row_x][2]
            gp2 = data[row_x_h][2]
            g = (gp1 - gp2) ** 2
            gammas.append(g)
        if len(gammas) == 0:
            gamma = 0
        else:
            gamma = np.sum(gammas) / (2 * len(gammas))
        smv.append([gamma, len(gammas)])

    # Selecting semivariances
    for i in range(len(lags)):
        if smv[i][0] > 0:
            semivariance.append([lags[i], smv[i][0], smv[i][1]])
        else:
            semivariance.append([lags[i], 0, 0])

    semivariance = np.vstack(semivariance)
    return semivariance


class Semivariance:

    def __init__(self, areal_data_file, lags=None, step_size=None, id_field='ID'):
        """Class for calculation of Areal semivariance. It us used by the methods of Area-to-Area and Area-to-Point
           Kriging. Class has two main methods: semivariance_from_centroids and semivariance_from_areal_data.
           Class is initialized by:
           areal_data: the dictionary of areas and their values,
           population_data: vector files with population centroids derived from population blocks for each area.
           To maintain stability areas in the areal_data and population blocks in the population_data must have
           the same ID filed - id_field, which allows algorithm to merge those datasets.

           Calculation methods of the class:
           C1. centroids_semivariance(lags=self.lags, step_size=self.step_size)

           Visualization methods of the class:


           Private methods of the class and their relations to the specific class variables and methods:


        """
        self.areal_data = areal_data_file
        self.geodataframe = gpd.read_file(areal_data_file)
        
        self.ranges = lags
        self.step = step_size

        self.id_field = id_field

        self.centroids = None  # variable updated by the centroids_semivariance() method
        self.distances_dict = None  # variable updated by the centroids_semivariance() method
        self.point_support_semivariance = None  # variable updated by the centroids_semivariance() method
        self.g_dict = {}

    def centroids_semivariance(self, lags=None, step_size=None, update=True, data_column='DATA'):
        """
        Function calculates semivariance of areal centroids and their values.
        :param lags: array of lags between points
        :param step_size: distance which should be included in the gamma parameter which enhances range of interest
        :param update: if True then class centroids and point_support_semivariance variables will be updated
        :param data_column: string with a name of column containing data values
        :return: semivariance: numpy array of pair of lag and semivariance values where
                 semivariance[0] = array of lags
                 semivariance[1] = array of lag's values
                 semivariance[2] = array of number of points in each lag
        """

        # Set lags
        if not lags:
            lags = self.ranges

        # Set step size
        if not step_size:
            step_size = self.step

        # Calculate centroids positions
        centroids = get_centroids(self.geodataframe, data_column, self.id_field, areal=True, dropna=True)

        # Calculate distances
        try:
            distance_array = calculate_distance(centroids[:, :-2])
        except TypeError:
            centroids = np.asarray(centroids)
            print('Given points array has been transformed into numpy array to calculate distance')
            distance_array = calculate_distance(centroids[:, :-2])

        semivariance = self._calculate_semivars(lags, step_size, centroids, distance_array)
        
        # Update object
        if update:
            self.centroids = centroids
            self.point_support_semivariance = semivariance

        return semivariance

    @staticmethod
    def _calculate_semivars(lags, step_size, points_array, distances_array, rate=None):
        """Method calculates semivariance.

        INPUT:
        :param lags: list of lags,
        :param step_size: step between lags,
        :param points_array: array with points and their values,
        :param distances_array: array with distances between points

        OUTPUT:
        :return semivariance: numpy array of pair of lag and semivariance values where
                 semivariance[0] = array of lags
                 semivariance[1] = array of lag's values
                 semivariance[2] = array of number of points in each lag"""
        smv = []
        semivariance = []

        # Calculate semivariances
        for h in lags:
            gammas = []
            distances_in_range = np.where(
                np.logical_and(
                    np.greater_equal(distances_array, h - step_size), np.less_equal(distances_array, h + step_size)))
            for i in range(0, len(distances_in_range[0])):
                row_x = distances_in_range[0][i]
                row_x_h = distances_in_range[1][i]
                if rate is not None:
                    gp1 = rate[0]
                    gp2 = rate[1]
                else:
                    gp1 = points_array[row_x][2]
                    gp2 = points_array[row_x_h][2]
                g = (gp1 - gp2) ** 2
                gammas.append(g)
            if len(gammas) == 0:
                gamma = 0
            else:
                gamma = np.sum(gammas) / (2 * len(gammas))
            smv.append([gamma, len(gammas)])

        # Selecting semivariances
        for i in range(len(lags)):
            if smv[i][0] > 0:
                semivariance.append([lags[i], smv[i][0], smv[i][1]])
            else:
                semivariance.append([lags[i], 0, 0])

        semivariance = np.vstack(semivariance)
        return semivariance
