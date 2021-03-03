from collections import OrderedDict

import numpy as np

from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.transform.select_values_in_range import select_values_in_range

import matplotlib.pyplot as plt


def build_variogram_point_cloud(data, step_size, max_range):
    """
    Function calculates variogram point cloud of a given set of points for
        a given set of distances. Variogram is calculated as a squared difference
        of each point against other point within range specified by step_size
        parameter. 

    INPUT:

    :param data: (numpy array) coordinates and their values,
    :param step_size: (float) step size of circle within radius are analyzed points,
    :param max_range: (float) maximum range of analysis.

    OUTPUT:

    :return: variogram_cloud - dict with pairs {lag: list of squared differences}.
    """

    distances = calc_point_to_point_distance(data[:, :-1])
    lags = np.arange(0, max_range, step_size)
    variogram_cloud = OrderedDict()

    # Calculate squared differences
    # They are halved to be compatibile with semivariogram

    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        sem = (data[distances_in_range[0], 2] - data[distances_in_range[1], 2]) ** 2
        sem = sem / 2
        variogram_cloud[h] = sem

    return variogram_cloud


def show_variogram_cloud(variogram_cloud, figsize=None):
    """
    Function shows boxplots of variogram lags. It is especially useful when
        you want to check outliers in your dataset.

    INPUT:

    :param variogram_cloud: (OrderedDict) lags and halved squared differences between
       points,
    :param figsize: (tuple) figure size (width, height).
    """
    if figsize is None:
        figsize = (14, 6)

    data = [x for x in variogram_cloud.values()]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data)
    xtick_names = plt.setp(ax, xticklabels=variogram_cloud.keys())
    plt.setp(xtick_names, rotation=45, fontsize=8)
    plt.show()


def calculate_semivariance(data, step_size, max_range):
    """Function calculates semivariance of a given set of points.

        Equation for calculation is:

            semivariance = 1 / (2 * N) * SUM(i=1, N) [z(x_i + h) - z(x_i)]^2

        where:

            - N - number of observation pairs,
            - h - distance (lag),
            - z(x_i) - value at location z_i,
            - (x_i + h) - location at a distance h from x_i.

        INPUT:

        :param data: (numpy array) coordinates and their values,
        :param step_size: (float) step size of circle within radius are analyzed points,
        :param max_range: (float) maximum range of analysis.

        OUTPUT:

        :return: semivariance: numpy array of pair of lag and semivariance values where:

            - semivariance[0] = array of lags,
            - semivariance[1] = array of lag's values,
            - semivariance[2] = array of number of points in each lag.
    """

    distances = calc_point_to_point_distance(data[:, :-1])

    semivariance = []

    lags = np.arange(0, max_range, step_size)

    # Calculate semivariances
    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        sem = (data[distances_in_range[0], 2] - data[distances_in_range[1], 2]) ** 2
        if len(sem) == 0:
            sem_value = 0
        else:
            sem_value = np.sum(sem) / (2 * len(sem))
        semivariance.append([h, sem_value, len(sem)])
    semivariance = np.vstack(semivariance)
    
    return semivariance


def calculate_weighted_semivariance(data, step_size, max_range):
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

        :param data: (numpy array) coordinates and their values and weights:
            [coordinate x, coordinate y, value, weight],
        :param step_size: (float) step size of circle within radius are analyzed points,
        :param max_range: (float) maximum range of analysis.


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

    lags = np.arange(0, max_range, step_size)

    # Calculate semivariances
    for idx, h in enumerate(lags):
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
        print('One or more of weights in dataset is set to 0, this may cause errors in the distance')


def calc_semivariance_from_pt_cloud(pt_cloud_dict):
    """
    Function calculates experimental semivariogram from point cloud variogram.

    INPUT:

    :param pt_cloud_dict: (OrderedDict) {lag: [values]}

    OUTPUT:
    :return: (numpy array) [lag, semivariance, number of points]
    """
    experimental_semivariogram = []
    for k in pt_cloud_dict.keys():
        try:
            experimental_semivariogram.append([k, np.mean(pt_cloud_dict[k]), len(pt_cloud_dict[k])])
        except ZeroDivisionError:
            # There are no points for this lag
            experimental_semivariogram.append([k, 0, 0])
    experimental_semivariogram = np.array(experimental_semivariogram)
    return experimental_semivariogram


def remove_outliers(data_dict, std_dist=2.):
    """Function removes outliers from each lag and returns dict without those values.

    INPUT:

    :param data_dict: (Ordered Dict) with {lag: list of values},
    :param std_dist: (float) number of standard deviations from the mean within values are passed.

    OUTPUT:

    :returns: (OrderedDict) {lag: [values]}"""

    output = OrderedDict()

    for lag in data_dict.keys():
        if isinstance(data_dict[lag], list):
            dd = np.array(data_dict[lag])
        else:
            dd = data_dict[lag].copy()

        mean_ = np.mean(dd)
        std_ = np.std(dd)
        upper_boundary = mean_ + std_dist * std_
        lower_boundary = mean_ - std_dist * std_

        vals = dd[(dd < upper_boundary) & (dd > lower_boundary)]
        output[lag] = vals

    return output
