from numpy.typing import ArrayLike
import os

import dask
import numpy as np
from tqdm import tqdm
from dask.diagnostics import ProgressBar

from pyinterpolate.kriging.point.ordinary import ok_calc
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.geo import geometry_and_values_array


def interpolate_points(
        theoretical_model: TheoreticalVariogram,
        unknown_locations: ArrayLike,
        known_locations: ArrayLike = None,
        known_values: ArrayLike = None,
        known_geometries: ArrayLike = None,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        progress_bar=True
) -> np.ndarray:
    """
    Function predicts values at unknown locations with Ordinary
    Kriging.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    unknown_locations : numpy array
        Points where you want to estimate value
        ``[(x, y), ...] <-> [(lon, lat), ...]``.

    known_locations : numpy array, optional
        The known locations: ``[x, y, value]``.

    known_values : ArrayLike, optional
        Observation in the i-th geometry (from ``known_geometries``). Optional
        parameter, if not given then ``known_locations`` must be provided.

    known_geometries : ArrayLike, optional
        Array or similar structure with geometries. It must have the same
        length as ``known_values``. Optional parameter, if not given then
        ``known_locations`` must be provided. Point type geometry.

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

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> from pyinterpolate import (build_experimental_variogram,
    ...     build_theoretical_variogram)
    >>> from pyinterpolate.core.pipelines.interpolate import interpolate_points
    >>>
    >>> dem = gpd.read_file('dem.gpkg')
    >>> unknown_locations = gpd.read_file('unknown_locations.gpkg')
    >>> step_size = 500
    >>> max_range = 10000
    >>> exp_variogram = build_experimental_variogram(
    ...     values=dem['dem'],
    ...     geometries=dem['geometry'],
    ...     step_size=step_size,
    ...     max_range=max_range
    ... )
    >>> theo_variogram = build_theoretical_variogram(exp_variogram)
    >>> interp = interpolate_points(
    ...     theoretical_model=theo_variogram,
    ...     unknown_locations=unknown_locations['geometry'],
    ...     known_values=dem['dem'],
    ...     known_geometries=dem['geometry']
    ... )
    >>> print(interp[0])
    [7.91222896e+01 9.72740449e+01 2.38012302e+05 5.51466805e+05]
    """

    if known_locations is None:
        known_locations = geometry_and_values_array(
            geometry=known_geometries,
            values=known_values
        )

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
        unknown_locations: ArrayLike,
        known_locations: ArrayLike = None,
        known_values: ArrayLike = None,
        known_geometries: ArrayLike = None,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        number_of_workers=1,
        progress_bar=True
) -> np.ndarray:
    """
    Function predicts values at unknown locations with Ordinary
    Kriging using Dask backend, makes sense when you must interpolate large
    number of points.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    unknown_locations : numpy array
        Points where you want to estimate value
        ``[(x, y), ...] <-> [(lon, lat), ...]``.

    known_locations : numpy array, optional
        The known locations: ``[x, y, value]``.

    known_values : ArrayLike, optional
        Observation in the i-th geometry (from ``known_geometries``). Optional
        parameter, if not given then ``known_locations`` must be provided.

    known_geometries : ArrayLike, optional
        Array or similar structure with geometries. It must have the same
        length as ``known_values``. Optional parameter, if not given then
        ``known_locations`` must be provided. Point type geometry.

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

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> from pyinterpolate import (build_experimental_variogram,
    ...     build_theoretical_variogram)
    >>> from pyinterpolate.core.pipelines.interpolate import interpolate_points_dask
    >>>
    >>> dem = gpd.read_file('dem.gpkg')
    >>> unknown_locations = gpd.read_file('unknown_locations.gpkg')
    >>> step_size = 500
    >>> max_range = 10000
    >>> exp_variogram = build_experimental_variogram(
    ...     values=dem['dem'],
    ...     geometries=dem['geometry'],
    ...     step_size=step_size,
    ...     max_range=max_range
    ... )
    >>> theo_variogram = build_theoretical_variogram(exp_variogram)
    >>> interp = interpolate_points_dask(
    ...     theoretical_model=theo_variogram,
    ...     unknown_locations=unknown_locations['geometry'],
    ...     known_values=dem['dem'],
    ...     known_geometries=dem['geometry']
    ... )
    >>> print(interp[0])
    [7.91222896e+01 9.72740449e+01 2.38012302e+05 5.51466805e+05]
    """

    if known_locations is None:
        known_locations = geometry_and_values_array(
            geometry=known_geometries,
            values=known_values
        )

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
                allow_approximate_solutions=allow_approximate_solutions
            )

            results.append(prediction)

        predictions = dask.delayed()(results)
        predictions = predictions.compute(num_workers=number_of_workers)
        return np.array(predictions)
