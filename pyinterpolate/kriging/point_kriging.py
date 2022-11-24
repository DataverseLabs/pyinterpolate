"""
Controls point kriging interpolation and parallelize it.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Union, List, Tuple

import os
import numpy as np
import dask
from dask.diagnostics import ProgressBar

from tqdm import tqdm

from pyinterpolate.kriging.models.point.ordinary_kriging import ordinary_kriging
from pyinterpolate.kriging.models.point.simple_kriging import simple_kriging
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram

pbar = ProgressBar()
pbar.register()


def kriging(observations: np.ndarray,
            theoretical_model: TheoreticalVariogram,
            points: Union[np.ndarray, List, Tuple],
            how: str = 'ok',
            neighbors_range: Union[float, None] = None,
            no_neighbors: int = 4,
            use_all_neighbors_in_range=False,
            sk_mean: Union[float, None] = None,
            allow_approx_solutions=False,
            number_of_workers: int = 1) -> np.ndarray:
    """Function manages Ordinary Kriging and Simple Kriging predictions.

    Parameters
    ----------
    observations : numpy array
        Known points and their values.

    theoretical_model : TheoreticalVariogram
        Fitted variogram model.

    points : numpy array
        Coordinates with missing values (to estimate results).

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

    number_of_workers : int, default=1
        How many processing units can be used for predictions. Increase it only for a very large number of
        interpolated points (~10k+).

    Returns
    -------
    : numpy array
        Predictions ``[predicted value, variance error, longitude (x), latitude (y)]``
    """

    # Check model type
    if how == 'sk' and sk_mean is None:
        raise AttributeError('You have chosen simple kriging "sk" as a baseline for your '
                             'interpolation but you did not set sk_mean parameter. You must do it '
                             'to perform calculations properly.')

    if number_of_workers == -1:
        core_num = os.cpu_count()
        if core_num > 1:
            number_of_workers = core_num - 1  # Safety reasons
        else:
            number_of_workers = core_num

    models = {'ok': ordinary_kriging,
              'sk': simple_kriging}

    if how not in list(models.keys()):
        raise KeyError(f'Given model not available, choose one from {list(models.keys())} instead.')
    else:
        model = models[how]

    results = []

    if number_of_workers == 1:
        # Don't use dask
        for point in tqdm(points):
            prediction = [np.nan, np.nan, np.nan, np.nan]
            if how == 'ok':
                prediction = model(
                    theoretical_model,
                    observations,
                    point,
                    neighbors_range=neighbors_range,
                    no_neighbors=no_neighbors,
                    use_all_neighbors_in_range=use_all_neighbors_in_range,
                    allow_approximate_solutions=allow_approx_solutions
                )
            elif how == 'sk':
                prediction = model(
                    theoretical_model,
                    observations,
                    point,
                    sk_mean,
                    neighbors_range=neighbors_range,
                    no_neighbors=no_neighbors,
                    use_all_neighbors_in_range=use_all_neighbors_in_range,
                    allow_approximate_solutions=allow_approx_solutions
                )
            results.append(prediction)
        predictions = np.array(results)
    else:
        # Use dask
        for point in points:
            prediction = [np.nan, np.nan, np.nan, np.nan]
            if how == 'ok':
                prediction = dask.delayed(model)(
                    theoretical_model,
                    observations,
                    point,
                    neighbors_range=neighbors_range,
                    no_neighbors=no_neighbors,
                    use_all_neighbors_in_range=use_all_neighbors_in_range,
                    allow_approximate_solutions=allow_approx_solutions
                )
            elif how == 'sk':
                prediction = dask.delayed(model)(
                    theoretical_model,
                    observations,
                    point,
                    sk_mean,
                    neighbors_range=neighbors_range,
                    no_neighbors=no_neighbors,
                    use_all_neighbors_in_range=use_all_neighbors_in_range,
                    allow_approximate_solutions=allow_approx_solutions
                )
            results.append(prediction)
        predictions = dask.delayed()(results)
        predictions = np.array(predictions.compute(num_workers=number_of_workers))
    return predictions
