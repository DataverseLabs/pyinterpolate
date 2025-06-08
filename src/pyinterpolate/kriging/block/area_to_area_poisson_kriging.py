"""
Area-to-area Poisson Kriging function.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

"""
from typing import Dict, Union, Hashable

import numpy as np

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.kriging.block.weights import pk_weights_array
from pyinterpolate.semivariogram.deconvolution.block_to_block_semivariance import weighted_avg_point_support_semivariances
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.select_poisson_kriging_data import select_poisson_kriging_data
from pyinterpolate.transform.statistical import sem_to_cov


def area_to_area_pk(semivariogram_model: TheoreticalVariogram,
                    point_support: PointSupport,
                    unknown_block_index: Union[str, Hashable],
                    number_of_neighbors: int,
                    neighbors_range: float = None,
                    raise_when_negative_prediction=True,
                    raise_when_negative_error=True) -> dict:
    """
    Function predicts areal value in an unknown location based on
    the area-to-area Poisson Kriging

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
        The maximum range for neighbors search. If not provided then
        it is read from the semivariogram model.

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    Returns
    -------
    results : Dict
        Block index, prediction, error: ``{"block_id", "zhat", "sig"}``

    Raises
    ------
    ValueError
        Raised when prediction or prediction error are negative.
    """
    # Prepare Kriging Data
    # {known block id: [
    #     (unknown x, unknown y),
    #     [unknown val, known val, distance between points]
    # ]}
    # Transform point support to dict
    # Kriging data
    kriging_data = select_poisson_kriging_data(
        block_index=unknown_block_index,
        point_support=point_support,
        semivariogram_model=semivariogram_model,
        number_of_neighbors=number_of_neighbors,
        neighbors_range=neighbors_range
    )

    closest_neighbors = kriging_data.neighbors_unique_indexes
    n = len(closest_neighbors)

    sill = semivariogram_model.sill

    # Semivariances between blocks
    avg_semivariances = kriging_data.weighted_b2b_semivariance()
    k = avg_semivariances.to_numpy()
    k = sem_to_cov(k, sill)
    covars = np.ones(len(k) + 1)
    covars[:-1] = k
    covars = covars.transpose()  # k_ones

    # Distances between neighboring areas
    # [
    #     (known block id a, known block id b),
    #     pt a val, pt b val, distance between points
    # ]
    distances_between_neighboring_point_supports = kriging_data.distances_between_neighboring_point_supports(
        point_support=point_support
    )

    # Semivariances between neighbors
    # (known block id a, known block id b) -> semivariance (Series)
    semivariances_between_neighboring_point_supports = weighted_avg_point_support_semivariances(
        semivariogram_model,
        distances_between_neighboring_point_supports,
        index_col='blocks_pair',
        val1_col='block_a_value',
        val2_col='block_b_value',
        dist_col='distance'
    )
    semivariances_nn = semivariances_between_neighboring_point_supports.to_numpy()
    predicted = sem_to_cov(semivariances_nn, sill)
    predicted = predicted.reshape((n, n))

    # Add diagonal weights
    block_values = point_support.blocks.get_blocks_values(
        indexes=closest_neighbors
    )
    totals = point_support.point_support_totals(blocks=closest_neighbors)
    p_weights = pk_weights_array(predicted.shape,
                                 block_values,
                                 totals)
    weighted_and_predicted = predicted + p_weights

    # Prepare weights matrix
    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[weighted_and_predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    try:
        w = np.linalg.solve(weights, covars)
    except TypeError:
        weights = weights.astype(float)
        covars = covars.astype(float)
        w = np.linalg.solve(weights, covars)

    zhat = block_values.dot(w[:-1])

    # Calculate prediction error
    if zhat < 0:
        if raise_when_negative_prediction:
            raise ValueError(f'Predicted value is {zhat} and it should '
                             f'not be lower than 0. Check your sampling '
                             f'grid, samples, number of neighbors or '
                             f'semivariogram model type.')

    sigmasq = np.matmul(w.T, covars)

    # Find covariance within unknown block's point support
    distances_within_unknown_block_point_support = kriging_data.distances_within_unknown_block(
        point_support=point_support
    )

    semivariance_within_unknown_block = weighted_avg_point_support_semivariances(
        semivariogram_model=semivariogram_model,
        distances_between_neighboring_point_supports=distances_within_unknown_block_point_support,
        index_col='block_id',
        val1_col='point_a_value',
        val2_col='point_b_value',
        dist_col='distance'
    )

    covariance_within_unknown = sem_to_cov(
        semivariance_within_unknown_block.to_numpy(),
        sill
    )

    sigmasq_2 = covariance_within_unknown[0] - sigmasq

    if sigmasq_2 < 0:
        if raise_when_negative_error:
            raise ValueError(f'Predicted error value is {sigmasq_2} and it '
                             f'should not be lower than 0. '
                             f'Check your sampling grid, samples, number of '
                             f'neighbors or semivariogram model type.')
        sigma = 0
    else:
        sigma = np.sqrt(sigmasq_2)

    results = {
        'block_id': unknown_block_index,
        'zhat': zhat,
        'sig': sigma
    }

    return results
