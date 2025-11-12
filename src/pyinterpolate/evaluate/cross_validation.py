from typing import Tuple, Union
from numpy.typing import ArrayLike

import numpy as np
from tqdm import tqdm

from pyinterpolate.semivariogram.theoretical.theoretical import TheoreticalVariogram
from pyinterpolate.kriging.point.ordinary import ordinary_kriging
from pyinterpolate.kriging.point.simple import simple_kriging
from pyinterpolate.transform.geo import geometry_and_values_array


def validate_kriging(
        theoretical_model: TheoreticalVariogram,
        points: ArrayLike = None,
        values: ArrayLike = None,
        geometries: ArrayLike = None,
        how: str = 'ok',
        neighbors_range: Union[float, None] = None,
        no_neighbors: int = 4,
        use_all_neighbors_in_range=False,
        sk_mean: Union[float, None] = None,
        allow_approximate_solutions=False,
        progress_bar: bool = True
) -> Tuple[float, float, np.ndarray]:
    """
    Function performs cross-validation of kriging models.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted variogram model.

    points : ArrayLike, optional
        Known points and their values ``[x, y, value]``.

    values : ArrayLike, optional
        Observation in the i-th geometry (from ``geometries``). Optional
        parameter, if not given then ``points`` must be provided.

    geometries : ArrayLike, optional
        Array or similar structure with geometries. It must have the same
        length as ``values``. Optional parameter, if not given then
        ``points`` must be provided. Point type geometry.

    how : str, default='ok'
        Select what kind of kriging you want to perform

          * 'ok': ordinary kriging,
          * 'sk': simple kriging - if it is set then ``sk_mean`` parameter
            must be provided.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is
        given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the number of neighbors within the
        ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then use all neighbors, do not
        clip their number.

    sk_mean : float, default=None
        The mean value of a process over a study area. Should be known
        before processing. That's why Simple Kriging has a limited number
        of applications. You must have multiple samples and well-known area to
        know this parameter.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on
        the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful
        when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    progress_bar : bool, default=True
        Show process status.

    Returns
    -------
    : Tuple
        Function returns tuple with:

          * Mean Prediction Error,
          * Mean Kriging Error: ratio of variance of prediction errors to
            the average variance error of kriging,
          * array with: ``[coordinate x, coordinate y, prediction error, kriging estimate error]``

    References
    ----------
    1. Clark, I., (2004), The Art of Cross Validation in Geostatistical
       Applications
    2. Clark I., (1979), Does Geostatistics Work, Proc. 16th APCOM,
       pp.213.-225.

    Examples
    --------
    >>> from pyinterpolate import (
    ...     ExperimentalVariogram,
    ...     validate_kriging,
    ...     TheoreticalVariogram
    ... )
    >>>
    >>>
    >>> POINTS_DATA = ...  # load dataset
    >>> POINTS_VARIOGRAM = ExperimentalVariogram(POINTS_DATA,
    ...                                          step_size=1,
    ...                                          max_range=6)
    >>> THEORETICAL_MODEL = TheoreticalVariogram()
    >>> THEORETICAL_MODEL.autofit(experimental_variogram=POINTS_VARIOGRAM,
    ...                           models_group='linear',
    ...                           nugget=0.0)
    >>> validation_results = validate_kriging(
    ...     theoretical_model=THEORETICAL_MODEL,
    ...     values=POINTS_DATA[:, -1],
    ...     geometries=POINTS_DATA[:, :-1],
    ...     no_neighbors=4,
    ...     progress_bar=False
    ... )
    >>> print(validation_results[0])  # mean prediction error
    -0.01613441673494531
    >>> print(validation_results[1])  # mean kriging error
    1.6386630811210166
    """
    # TODO:
    # Use (2) to calc Z-score
    # TODO:
    # Validation tutorials
    # TODO:
    # Areal kriging validation
    # Initialize array for coordinates and errors
    coordinates_and_errors = []

    if points is None:
        points = geometry_and_values_array(
            geometry=geometries,
            values=values
        )

    # Divide observations
    for idx, row in enumerate(tqdm(points, disable=not progress_bar)):
        clipped_point = row[:-1]
        data_points = np.delete(points, idx, 0)

        if how == 'ok':
            preds = ordinary_kriging(
                theoretical_model=theoretical_model,
                known_locations=data_points,
                unknown_locations=clipped_point,
                neighbors_range=neighbors_range,
                no_neighbors=no_neighbors,
                use_all_neighbors_in_range=use_all_neighbors_in_range,
                allow_approximate_solutions=allow_approximate_solutions,
                progress_bar=False
            )
        elif how == 'sk':
            preds = simple_kriging(
                theoretical_model=theoretical_model,
                known_locations=data_points,
                process_mean=sk_mean,
                unknown_locations=clipped_point,
                neighbors_range=neighbors_range,
                no_neighbors=no_neighbors,
                use_all_neighbors_in_range=use_all_neighbors_in_range,
                allow_approximate_solutions=allow_approximate_solutions,
                progress_bar=False
            )
        else:
            raise KeyError(
                'Allowed kriging types (parameter "how") are:'
                ' "ok" - ordinary kriging,'
                ' and "sk" - simple kriging.'
            )

        if len(preds) == 1:
            preds = preds[0]

        prediction_error = row[-1] - preds[0]

        coordinates_and_errors.append(
            [preds[2], preds[3], prediction_error, preds[1]]
        )

    output_arr = np.array(coordinates_and_errors)
    mean_prediction_error = np.mean(output_arr[:, 2])
    mean_variance_error = np.var(output_arr[:, 2]) / np.mean(output_arr[:, 3])

    return mean_prediction_error, mean_variance_error, output_arr
