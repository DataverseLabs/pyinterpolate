"""
Centroid-based Poisson Kriging function.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Union, Hashable, Dict

import numpy as np

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.kriging.block.weights import pk_weights_array
from pyinterpolate.kriging.utils.point_kriging_solve import solve_weights
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.select_poisson_kriging_data import select_centroid_poisson_kriging_data
from pyinterpolate.transform.statistical import sem_to_cov


def centroid_poisson_kriging(semivariogram_model: TheoreticalVariogram,
                             point_support: PointSupport,
                             unknown_block_index: Union[str, Hashable],
                             number_of_neighbors: int,
                             neighbors_range: float = None,
                             is_weighted_by_point_support=True,
                             raise_when_negative_prediction=True,
                             raise_when_negative_error=True,
                             allow_lsa=False) -> Dict:
    """
    Function performs centroid-based Poisson Kriging of blocks (areal) data.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        A fitted variogram.

    point_support : PointSupport
        The point support of polygons.

    unknown_block_index : Hashable
        The id of the block with the unknown value.

    number_of_neighbors : int
        The minimum number of neighbours that can potentially affect
        the unknown block.

    neighbors_range : float, optional
        The maximum range for neighbors search. If not provided then it is
        read from the semivariogram model.

    is_weighted_by_point_support : bool, default = True
        Are distances between blocks weighted by the point support?

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    allow_lsa : bool, default=False
        Allows the approximation of kriging weights based on the OLS
        algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful
        when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    results : Dict
        Block index, prediction, error: ``{"block id", "zhat", "sig"}``

    Raises
    ------
    ValueError
        Raised when prediction or prediction error are negative.
    """
    # Kriging data
    kriging_data = select_centroid_poisson_kriging_data(
        block_index=unknown_block_index,
        point_support=point_support,
        semivariogram_model=semivariogram_model,
        number_of_neighbors=number_of_neighbors,
        weighted=is_weighted_by_point_support,
        neighbors_range=neighbors_range
    )

    sill = semivariogram_model.sill
    n = len(kriging_data)

    distances = kriging_data.distances
    values = kriging_data.values
    kindexes = kriging_data.neighbors_indexes

    partial_semivars = semivariogram_model.predict(distances)
    pcovars = sem_to_cov(partial_semivars, sill)
    covars = np.ones(len(pcovars) + 1)
    covars[:-1] = pcovars
    covars = covars.transpose()

    # Distances between known blocks
    block_distances = point_support.get_distances_between_known_blocks(
        block_ids=kindexes
    )
    known_blocks_semivars = semivariogram_model.predict(
        block_distances.flatten()
    )
    predicted = np.array(known_blocks_semivars.reshape(n, n))
    predicted = sem_to_cov(predicted, sill)

    # Add diagonal weights to predicted semivars array
    totals = point_support.point_support_totals(
        kindexes
    )
    weights = pk_weights_array(predicted.shape,
                               values,
                               totals)
    weighted_and_predicted = predicted + weights

    # Prepare matrix for solving kriging system
    ones_col = np.ones((weighted_and_predicted.shape[0], 1))
    weighted_and_predicted = np.c_[weighted_and_predicted, ones_col]
    ones_row = np.ones((1, weighted_and_predicted.shape[1]))
    ones_row[0][-1] = 0
    kriging_weights = np.r_[weighted_and_predicted, ones_row]

    # Solve Kriging system
    try:
        output_weights = solve_weights(weights=kriging_weights,
                                       k=covars,
                                       allow_lsa=allow_lsa)
    except np.linalg.LinAlgError as _:
        msg = ('Singular matrix in Kriging system detected, check if you '
               'have duplicated coordinates in the input dataset.')
        raise RuntimeError(msg)

    zhat = values.dot(output_weights[:-1])

    if zhat < 0:
        if raise_when_negative_prediction:
            raise ValueError(f'Predicted value is {zhat} and it should not '
                             f'be lower than 0. Check your sampling '
                             f'grid, samples, number of neighbors or '
                             f'semivariogram model type.')

    sigmasq = np.matmul(output_weights.T, covars)

    if sigmasq < 0:
        if raise_when_negative_error:
            raise ValueError(f'Predicted error value is {sigmasq} and it '
                             f'should not be lower than 0. '
                             f'Check your sampling grid, samples, number of '
                             f'neighbors or semivariogram model type.'
                             f'Error is related to the block with index '
                             f'{unknown_block_index}')
        sigma = np.nan
    else:
        sigma = np.sqrt(sigmasq)

    # Prepare output
    results = {
        'block_id': unknown_block_index,
        'zhat': zhat,
        'sig': sigma
    }
    return results
