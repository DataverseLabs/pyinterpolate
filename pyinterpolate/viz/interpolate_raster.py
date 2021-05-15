import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.kriging.point_kriging.kriging import Krige


def show_data(data_matrix):
    plt.figure()
    plt.imshow(data_matrix, cmap='Spectral_r')
    plt.title('Interpolated dataset')
    plt.colorbar()
    plt.show()


def _set_dims(xs, ys, dmax):
    """
    Function sets dimensions of the output array.
    :param xs: (array) all x values,
    :param ys: (array) all y values,
    :param dmax: (int) max dimension,
    :return: x_dim_coords, y_dim_coords, [properties]
    """

    xmin = np.min(xs)
    xmax = np.max(xs)

    ymin = np.min(ys)
    ymax = np.max(ys)

    x_abs = np.abs(xmax - xmin)
    y_abs = np.abs(ymax - ymin)

    if x_abs > y_abs:
        step = x_abs / dmax
        x_dim_coords = np.arange(xmin + step, xmax + step, step)
        y_dim_coords = np.arange(ymin + step, ymax + step, step)
    else:
        step = y_abs / dmax
        y_dim_coords = np.arange(ymin + step, ymax + step, step)
        x_dim_coords = np.arange(xmin + step, xmax + step, step)

    # y_dim_coords must be flipped

    y_dim_coords = y_dim_coords[::-1]
    return x_dim_coords, y_dim_coords, [step, xmin, xmax, ymin, ymax]


def update_interpolation_matrix(rows_coords, cols_coords, kriging_model, no_of_neighbors):
    output_vals = np.zeros(shape=(len(rows_coords), len(cols_coords)))
    output_errs = np.zeros(shape=(len(rows_coords), len(cols_coords)))

    for ridx, point_row in enumerate(tqdm(rows_coords)):
        for cidx, point_col in enumerate(cols_coords):
            predicted = kriging_model.ordinary_kriging(
                [point_col, point_row], no_of_neighbors, False
            )
            output_vals[ridx, cidx] = predicted[0]
            output_errs[ridx, cidx] = predicted[1]

    return output_vals, output_errs


def interpolate_raster(data, dim=1000, number_of_neighbors=4, semivariogram_model=None):
    """
    Function interpolates raster from data points using ordinary kriging.

    INPUT:

    :param data: (numpy array / list) [coordinate x, coordinate y, value],
    :param dim: (int) number of pixels (points) of a larger dimension (it could be width or height),
    :param number_of_neighbors: (int) default=16, number of points used to interpolate data,
    :param semivariogram_model: (TheoreticalSemivariance) default=None, Theoretical Semivariogram model,
        if not provided then it is estimated from a given dataset.

    OUTPUT:

    :return: (numpy array) [numpy array of interpolated values, numpy array of interpolation errors,
        [pixel size, min x, max x, min y, max y]]
    """

    # Set dimension

    if isinstance(data, list):
        data = np.array(data)

    cols_coords, rows_coords, props = _set_dims(data[:, 0], data[:, 1], dim)

    # Calculate semivariance if not provided

    if semivariogram_model is None:
        distances = calc_point_to_point_distance(data[:, :-1])

        maximum_range = np.max(distances)
        number_of_divisions = 100
        step_size = maximum_range / number_of_divisions

        semivariance = calculate_semivariance(data, step_size, maximum_range)

        ts = TheoreticalSemivariogram(data, semivariance, False)
        ts.find_optimal_model(False, number_of_neighbors)
    else:
        ts = semivariogram_model

    # Interpolate data point by point

    k = Krige(ts, data)

    kriged_matrix, kriged_errors = update_interpolation_matrix(rows_coords, cols_coords,
                                                               k, number_of_neighbors)

    return [kriged_matrix, kriged_errors], props
