import numpy as np
from pyinterpolate.data_processing.data_preparation.read_data import read_point_data
from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_weighted_semivariance


def test_calculate_semivariance():
    path = 'sample_data/poland_dem_gorzow_wielkopolski'

    dataset = read_point_data(path, 'txt')

    # Set semivariance params
    distances = calc_point_to_point_distance(dataset[:, :-1])

    maximum_range = np.max(distances)
    number_of_divisions = 10
    step_size = maximum_range / number_of_divisions
    lags = np.arange(0, maximum_range, step_size)

    gamma = calculate_semivariance(dataset, lags, step_size)

    output_int = [115, 258, 419, 538, 572, 547, 530, 563, 613, 583]

    assert (gamma[:, 1].astype(np.int) == np.array(output_int)).all()


def test_calculate_weighted_semivariance():
    path = 'sample_data/poland_dem_gorzow_wielkopolski'

    dataset = read_point_data(path, 'txt')
    new_col = np.arange(1, len(dataset) + 1)

    dataset_weights = np.zeros((dataset.shape[0], dataset.shape[1] + 1))
    dataset_weights[:, :-1] = dataset
    dataset_weights[:, -1] = new_col

    # Set semivariance params
    distances = calc_point_to_point_distance(dataset_weights[:, :-2])

    maximum_range = np.max(distances)
    number_of_divisions = 10
    step_size = maximum_range / number_of_divisions
    lags = np.arange(0, maximum_range, step_size)

    gamma_w = calculate_weighted_semivariance(dataset_weights, lags, step_size)

    output_int = [105, 237, 385, 497, 526, 491, 454, 460, 499, 464]

    assert (gamma_w[:, 1].astype(np.int) == np.array(output_int)).all()


if __name__ == '__main__':
    test_calculate_semivariance()
    test_calculate_weighted_semivariance()