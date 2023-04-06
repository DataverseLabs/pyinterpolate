from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from pyinterpolate import TheoreticalVariogram
from pyinterpolate.kriging.point_kriging import kriging


def validate_kriging(points: np.ndarray,
                     theoretical_model: TheoreticalVariogram,
                     how: str = 'ok',
                     neighbors_range: Union[float, None] = None,
                     no_neighbors: int = 4,
                     use_all_neighbors_in_range=False,
                     sk_mean: Union[float, None] = None,
                     allow_approx_solutions=False) -> Tuple[float, float, np.ndarray]:
    """
    Function performs cross-validation of kriging models.

    Parameters
    ----------
    points : numpy array
        Known points and their values.

    theoretical_model : TheoreticalVariogram
        Fitted variogram model.

    how : str, default='ok'
        Select what kind of kriging you want to perform:
          * 'ok': ordinary kriging,
          * 'sk': simple kriging - if it is set then ``sk_mean`` parameter must be provided.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    sk_mean : float, default=None
        The mean value of a process over a study area. Should be know before processing. That's why Simple
        Kriging has a limited number of applications. You must have multiple samples and well-known area to
        know this parameter.

    allow_approx_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    : Tuple
        Function returns tuple with:
          * Mean Prediction Error,
          * Mean Kriging Error: ratio of variance of prediction errors to the average variance error of kriging,
          * array with: ``[coordinate x, coordinate y, prediction error, kriging estimate error]``

    References
    ----------
    1. Clark, I., (2004) “The Art of Cross Validation in Geostatistical Applications"
    2. Clark I., (1979) "Does Geostatistics Work", Proc. 16th APCOM, pp.213.-225.
    """
    # TODO:
    # Use (2) to calc Z-score
    # TODO:
    # Validation tutorials
    # TODO:
    # Areal kriging validation
    # Initialize array for coordinates and errors
    coordinates_and_errors = []

    # Divide observations
    for idx, row in enumerate(tqdm(points)):
        obs_coordinates = [row[:-1]]
        other_rows = np.delete(points, idx, 0)

        preds = kriging(
            observations=other_rows,
            theoretical_model=theoretical_model,
            points=obs_coordinates,
            how=how,
            neighbors_range=neighbors_range,
            no_neighbors=no_neighbors,
            use_all_neighbors_in_range=use_all_neighbors_in_range,
            sk_mean=sk_mean,
            allow_approx_solutions=allow_approx_solutions,
            number_of_workers=1,
            show_progress_bar=False
        )

        prediction_error = row[-1] - preds[0][0]

        coordinates_and_errors.append(
            [obs_coordinates[0][0], obs_coordinates[0][1], prediction_error, preds[0][1]]
        )

    output_arr = np.array(coordinates_and_errors)
    mean_prediction_error = np.mean(output_arr[:, 2])
    mean_variance_error = np.var(output_arr[:, 2]) / np.mean(output_arr[:, 3])

    return mean_prediction_error, mean_variance_error, output_arr
