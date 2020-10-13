import numpy as np

from tqdm import tqdm

from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance
from pyinterpolate.data_processing.data_preparation.select_values_in_range import select_values_in_range


def calculate_semivariance(data, lags, step_size):
    """Function calculates semivariance of a given set of points.

        Equation for calculation is:

            semivariance = 1 / (2 * N) * SUM(i=1, N) [z(x_i + h) - z(x_i)]^2

        where:

            - N - number of observation pairs,
            - h - distance (lag),
            - z(x_i) - value at location z_i,
            - (x_i + h) - location at a distance h from x_i.

        INPUT:

        :param data: array of coordinates and their values,
        :param lags: array of lags between points,
        :param step_size: distance which should be included in the gamma parameter which enhances range of interest.

        OUTPUT:

        :return: semivariance: numpy array of pair of lag and semivariance values where:

            - semivariance[0] = array of lags,
            - semivariance[1] = array of lag's values,
            - semivariance[2] = array of number of points in each lag.
    """

    distances = calc_point_to_point_distance(data[:, :-1])

    semivariance = []

    # Calculate semivariances
    for h in tqdm(lags):
        distances_in_range = select_values_in_range(distances, h, step_size)
        sem = (data[distances_in_range[0], 2] - data[distances_in_range[1], 2]) ** 2
        if len(sem) == 0:
            sem_value = 0
        else:
            sem_value = np.sum(sem) / (2 * len(sem))
        semivariance.append([h, sem_value, len(sem)])
    semivariance = np.vstack(semivariance)
    
    return semivariance


def calculate_weighted_semivariance(data, lags, step_size):
    """Function calculates weighted semivariance following Monestiez et al.:

        A. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C: Comparison of model based geostatistical methods
        in ecology: application to fin whale spatial distribution in northwestern Mediterranean Sea.
        In Geostatistics Banff 2004 Volume 2. Edited by: Leuangthong O, Deutsch CV. Dordrecht, The Netherlands,
        Kluwer Academic Publishers; 2005:777-786.

        B. Monestiez P, Dubroca L, Bonnin E, Durbec JP, Guinet C: Geostatistical modelling of spatial distribution
        of Balenoptera physalus in the northwestern Mediterranean Sea from sparse count data and heterogeneous
        observation efforts. Ecological Modelling 2006 in press.

        Equation for calculation is:

        s(h) = [1 / (2 * SUM(a=1, N(h)) (n(u_a) * n(u_a + h)) /...
                    / (n(u_a) + n(u_a + h)))] *...
                    * SUM(a=1, N(h)) {[(n(u_a) * n(u_a + h)) / (n(u_a) + n(u_a + h))] * [(z(u_a) - z(u_a + h))^2] - m'}

        where:

        - s(h) - Semivariogram of the risk,
        - n(u_a) - size of the population at risk in the unit a,
        - z(u_a) - mortality rate at the unit a,
        - u_a + h - area at the distance (h) from the analyzed area,
        - m' - population weighted mean of rates.

        INPUT:

        :param data: (numpy array) [coordinate x, coordinate y, value, weight],
        :param lags: (array) of lags [lag1, lag2, lag...]
        :param step_size: step size of search radius.


        OUTPUT:

        :return: semivariance: numpy array of pair of lag and semivariance values where:

            - semivariance[0] = array of lags
            - semivariance[1] = array of lag's values
            - semivariance[2] = array of number of points in each lag.
    """

    # TEST: test if any 0-weight is inside the dataset

    _test_weights(data[:, -1])

    # Calculate distance

    distances = calc_point_to_point_distance(data[:, :-2])

    # Prepare semivariance arrays
    smv = []
    semivariance = []

    # Calculate semivariances
    for idx, h in enumerate(tqdm(lags)):
        distances_in_range = select_values_in_range(distances, h, step_size)

        # Weights
        weight1 = data[distances_in_range[0], 3]
        weight2 = data[distances_in_range[1], 3]

        weights = (weight1 * weight2) / (weight1 + weight2)
        weights_sum = np.sum(weights)

        # Values
        val1 = data[distances_in_range[0], 2]
        val2 = data[distances_in_range[1], 2]

        # Weighted mean of values
        weighted_mean = ((weight1 * val1) + (weight2 * val2)) / weights_sum

        sem = weights * (data[distances_in_range[0], 2] - data[distances_in_range[1], 2]) ** 2
        sem_value = (np.sum(sem) - np.sum(weighted_mean)) / (2 * np.sum(weights_sum))
        smv.append([sem_value, len(sem)])
        if smv[idx][0] > 0:
            semivariance.append([h, smv[idx][0], smv[idx][1]])
        else:
            semivariance.append([h, 0, 0])

    semivariance = np.vstack(semivariance)
    return semivariance


def _test_weights(arr):
    if 0 in arr:
        print('One or more of weights in dataset is set to 0, this may cause errors in the calculations')
