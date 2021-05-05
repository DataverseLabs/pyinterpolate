from collections import OrderedDict

import numpy as np

from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.transform.select_values_in_range import select_values_in_range, check_points_within_ellipse

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

        s(h) = [1 / 2 * SUM|a=1, N(h)|: w(0)] ...
                    * [SUM|a=1, N(h)|: (w(0) * [(z(u_a) - z(u_a + h))^2] - m']

        where:

        w(0) = (n(u_a) * n(u_a + h)) / ((n(u_a) + n(u_a + h))

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

        # Any distance in a range?
        if len(distances_in_range[0]) > 0:
            # Weights
            ws_a = data[distances_in_range[0], 3]
            ws_ah = data[distances_in_range[1], 3]

            weights = (ws_a * ws_ah) / (ws_a + ws_ah)

            # Values
            z_a = data[distances_in_range[0], 2]
            z_ah = data[distances_in_range[1], 2]
            z_err = (z_a - z_ah) ** 2

            # m'
            _vals = data[distances_in_range[0], 2]
            _weights = data[distances_in_range[0], 3]
            mweighted_mean = np.average(_vals,
                                        weights=_weights)

            # numerator: w(0) * [(z(u_a) - z(u_a + h))^2] - m'
            numerator = weights * z_err - mweighted_mean

            # semivariance
            sem_value = 0.5 * (np.sum(numerator) / np.sum(weights))
            semivariance.append([h, sem_value, len(numerator)])
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


def calculate_directional_semivariogram(data, step_size, max_range, direction=0, tolerance=0.1):
    """
    Function calculates directional semivariogram of points.

    Semivariance is calculated as:

    semivar = 1/2*N SUM(from i=1 to N){[z(x_i + h) - z(x_i)]^2}

    where:
    N - number of observation pairs,
    h - distance (lag),
    z(xi) - value at location zi,
    (xi+h) - location at a distance h from xi.

    INPUT:

    :param data: (numpy array) coordinates and their values,
    :param step_size: (float) distance between lags within each points are included in the calculations,
    :param max_range: (float) maximum range of analysis,
    :param direction: (float) direction of semivariogram, values from 0 to 360 degrees:
        0 or 180: is NS direction,
        90 or 270 is EW direction,
        30 or 210 is NE-SW direction,
        120 or 300 is NW-SE direction,
    :param tolerance: (float) value in range (0-1) normalized to [0 : 0.5] to select tolerance of semivariogram. If tolerance
        is 0 then points must be placed at a single line with beginning in the origin of coordinate system and angle
        given by y axis and direction parameter. If tolerance is greater than 0 then semivariance is estimated
        from elliptical area with major axis with the same direction as the line for 0 tolerance and minor axis
        of a size:

        (tolerance * step_size)

        and major axis (pointed in NS direction):

        ((1 - tolerance) * step_size)

        and baseline point at a center of ellipse. Tolerance == 1 (normalized to 0.5) creates omnidirectional
        semivariogram.

    OUTPUT:

    :return: (numpy array) semivariance - array of pair of lag and semivariance values where:

    semivariance[0] = array of lags;
    semivariance[1] = array of lag's values;
    semivariance[2] = array of number of points in each lag.

    """

    if isinstance(data, list):
        data = np.array(data)

    if tolerance <= 0 or tolerance > 1:
        raise ValueError('Parameter tolerance should be set in the range (0, 1] to avoid undefined behavior')

    if direction < 0 or direction > 360:
        raise ValueError('Parameter direction should be set in range [0, 360].')

    if tolerance == 1:

        semivariances = calculate_semivariance(data, step_size, max_range)
        return semivariances
    else:
        angle = (np.pi / 180) * direction
        lags = np.arange(0, max_range, step_size)

        semivariances = [[0, 0, 0]]  # First, zero lag

        previous_lag = 0
        for lag in lags[1:]:
            semivars_per_lag = []
            for point in data:
                mask = check_points_within_ellipse(point, data, lag, previous_lag, angle, tolerance)

                pts_in_range = data[mask, -1]

                # Calculate semivars

                if len(pts_in_range) > 0:
                    semivars = (pts_in_range - point[-1]) ** 2
                    semivars_per_lag.extend(semivars)
                else:
                    semivars_per_lag.append(0)

            semivariance = np.mean(semivars_per_lag) / 2
            semivariances.append([lag, semivariance, len(semivars_per_lag)])
            previous_lag = lag

        return np.array(semivariances)