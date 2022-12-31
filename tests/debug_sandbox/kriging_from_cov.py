# Imports
from typing import List, Tuple, Union
import numpy as np
from pyinterpolate import read_txt
from pyinterpolate import ordinary_kriging  # kriging models
from pyinterpolate.kriging.utils.process import get_predictions, solve_weights
from pyinterpolate.variogram.empirical.experimental_variogram import ExperimentalVariogram
from pyinterpolate import TheoreticalVariogram

# Read data
dem = read_txt('../samples/point_data/txt/pl_dem_epsg2180.txt')


def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1), int(frac * len(dataset)), replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


known_points, unknown_points = create_model_validation_sets(dem)


def sem_to_cov(semivariances, sill) -> np.ndarray:
    """
    Function transforms semivariances into a covariances.

    Parameters
    ----------
    semivariances : Iterable

    sill : float

    Returns
    -------
    covariances : numpy array

    Notes
    -----

    sem = sill - cov
    cov = sill - sem
    """

    if isinstance(semivariances, np.ndarray):
        return sill - semivariances

    return sill - np.asarray(semivariances)


def ordinary_kriging2(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
        sill,
        neighbors_range=None,
        no_neighbors=4,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False
) -> List:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        A trained theoretical variogram model.

    known_locations : numpy array
        The known locations.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``

    Raises
    ------
    RunetimeError
        Singularity matrix in a Kriging system.
    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            no_neighbors,
                                            use_all_neighbors_in_range)

    k = sem_to_cov(k, sill)
    predicted = sem_to_cov(predicted, sill)

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    try:
        output_weights = solve_weights(weights, k, allow_approximate_solutions)
    except np.linalg.LinAlgError as _:
        msg = 'Singular matrix in Kriging system detected, check if you have duplicated coordinates ' \
              'in the ``known_locations`` variable.'
        raise RuntimeError(msg)

    zhat = dataset[:, -2].dot(output_weights[:-1])

    sigma = sill - np.matmul(output_weights.T, k)

    if sigma < 0:
        return [zhat, np.nan, unknown_location[0], unknown_location[1]]

    return [zhat, sigma, unknown_location[0], unknown_location[1]]


if __name__ == '__main__':

    exp_var = ExperimentalVariogram(input_array=known_points, step_size=500, max_range=20000)
    theo_var = TheoreticalVariogram()
    theo_var.autofit(exp_var, model_types='spherical')
    # theo_var.plot()

    for _unknown_pt in unknown_points:
        predicted_sem = ordinary_kriging(
            theoretical_model=theo_var,
            known_locations=known_points,
            unknown_location=_unknown_pt[:-1]
        )
        predicted_cov = ordinary_kriging2(
            theoretical_model=theo_var,
            known_locations=known_points,
            unknown_location=_unknown_pt[:-1],
            sill=exp_var.variance
        )

        assert np.allclose(predicted_cov, predicted_sem, rtol=10, equal_nan=True)
