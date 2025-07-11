import os

import dask
import numpy as np
from tqdm import tqdm
from dask.diagnostics import ProgressBar

from pyinterpolate.kriging.point.ordinary import ok_calc
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
    Function predicts values at unknown locations with Ordinary
    Kriging.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        The known locations: x, y, value.

    unknown_locations : numpy array
        Points where you want to estimate value
        ``[(x, y), ...] <-> [(lon, lat), ...]``.

    neighbors_range : float, default=None
        The maximum distance where we search for the neighbors.
        If ``None`` is given then range is selected from
        the theoretical model's ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        Maximum number of degrees for neighbors search angle.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the
        ``neighbors_range`` is greater than the ``number_of_neighbors``
        parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on
        the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be
        useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation. But the
        better idea is to get rid of those clusters.

    progress_bar : bool, default = True
        Shows progress bar

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``
    """

    interpolated_results = []

    _disable_progress_bar = not progress_bar

    for upoints in tqdm(unknown_locations, disable=_disable_progress_bar):
        res = ok_calc(
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
            res
        )

    return np.array(interpolated_results)


def interpolate_points_dask(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_locations: np.ndarray,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        number_of_workers=1,
        progress_bar=True
):
    """
    Function predicts values at unknown locations with Ordinary
    Kriging using Dask backend, makes sense when you must interpolate large
    number of points.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        The known locations: x, y, value.

    unknown_locations : numpy array
        Points where you want to estimate value
        ``[(x, y), ...] <-> [(lon, lat), ...]``.

    neighbors_range : float, default=None
        The maximum distance where we search for the neighbors.
        If ``None`` is given then range is selected from
        the theoretical model's ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        Maximum number of degrees for neighbors search angle.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the
        ``neighbors_range`` is greater than the ``number_of_neighbors``
        parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on
        the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be
        useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation. But the
        better idea is to get rid of those clusters.

    number_of_workers : int, default = 1
        How many processing units can be used for predictions.
        Increase it only for a very large number of
        interpolated points (~10k+).

    progress_bar : bool, default = True
        Shows progress bar

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``
    """

    if number_of_workers == -1:
        core_num = os.cpu_count()
        if core_num > 1:
            number_of_workers = core_num - 1  # Safety reasons
        else:
            number_of_workers = core_num

    if number_of_workers == 1:
        results = interpolate_points(
            theoretical_model=theoretical_model,
            known_locations=known_locations,
            unknown_locations=unknown_locations,
            neighbors_range=neighbors_range,
            no_neighbors=no_neighbors,
            max_tick=max_tick,
            use_all_neighbors_in_range=use_all_neighbors_in_range,
            allow_approximate_solutions=allow_approximate_solutions,
            progress_bar=progress_bar
        )
        return results
    else:
        pbar = ProgressBar()
        pbar.register()

        results = []
        for upoints in unknown_locations:
            prediction = dask.delayed(ok_calc)(
                theoretical_model=theoretical_model,
                known_locations=known_locations,
                unknown_location=upoints,
                neighbors_range=neighbors_range,
                no_neighbors=no_neighbors,
                max_tick=max_tick,
                use_all_neighbors_in_range=use_all_neighbors_in_range,
                allow_approximate_solutions=allow_approximate_solutions,
                progress_bar=False
            )

            results.append(prediction)

        predictions = dask.delayed()(results)
        predictions = predictions.compute(num_workers=number_of_workers)
        return np.array(predictions)
