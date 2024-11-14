import numpy as np
from tqdm import tqdm

from pyinterpolate.kriging.point.ordinary import ordinary_kriging
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def interpolate_points(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_locations: np.ndarray,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        progress_bar=True
):
    """
    Function predicts values at unknown locations with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        The known locations.

    unknown_locations : numpy array
        Points where you want to estimate value ``[(x, y), ...] <-> [(lon, lat), ...]``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how big should be a tolerance for increasing
        the search angle (how many degrees more).

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging custom_weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    progress_bar : bool, default = True
        Shows progress bar

    Returns
    -------
    : numpy array
        ``[[longitude (x), latitude (y), predicted value, variance error]]``
    """

    interpolated_results = []

    for upoints in tqdm(unknown_locations, disable=not(progress_bar)):
        res = ordinary_kriging(
            theoretical_model=theoretical_model,
            known_locations=known_locations,
            unknown_location=upoints,
            neighbors_range=neighbors_range,
            no_neighbors=no_neighbors,
            max_tick=max_tick,
            use_all_neighbors_in_range=use_all_neighbors_in_range,
            allow_approximate_solutions=allow_approximate_solutions
        )

        interpolated_results.append(
            [res[0], res[1]]
        )

    return np.array(interpolated_results)
