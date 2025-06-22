"""
Area-to-point Poisson Kriging function.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Union, Hashable

import numpy as np

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.kriging.block.weights import pk_weights_array
from pyinterpolate.semivariogram.deconvolution.block_to_block_semivariance import \
    weighted_avg_point_support_semivariances
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.select_poisson_kriging_data import select_poisson_kriging_data
from pyinterpolate.transform.statistical import sem_to_cov
from pyinterpolate.transform.transform import add_ones


def area_to_point_pk(semivariogram_model: TheoreticalVariogram,
                     point_support: PointSupport,
                     unknown_block_index: Union[str, Hashable],
                     number_of_neighbors: int,
                     neighbors_range: float = None,
                     raise_when_negative_prediction=True,
                     raise_when_negative_error=True,
                     err_to_nan=True):
    """
    Function predicts point-support value in the unknown location based on
    the area-to-point Poisson Kriging

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

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    err_to_nan : bool, default=True
        When point interpolation returns ``ValueError`` then set prediction
        or variance error to ``NaN``.

    Returns
    -------
    results : Dict[numpy array]
        ```{"block_id": [["x", "y", "zhat", "sig"], ...]}``

    Raises
    ------
    ValueError
        Prediction or prediction error are negative.
    """
    # Prepare Kriging Data
    # {
    #    known block id: [
    #        (unknown x, unknown y),
    #        [unknown val, known val, distance between points]
    #    ]
    # }

    kriging_data = select_poisson_kriging_data(
        block_index=unknown_block_index,
        point_support=point_support,
        semivariogram_model=semivariogram_model,
        number_of_neighbors=number_of_neighbors,
        neighbors_range=neighbors_range
    )

    closest_neighbors = kriging_data.neighbors_unique_indexes
    poss_neighbors = point_support.no_possible_neighbors

    if poss_neighbors > 0:
        if poss_neighbors < len(closest_neighbors):
            # todo? Move this step into select_poisson_kriging_data
            closest_neighbors = closest_neighbors[:poss_neighbors]
            kriging_data._neighbors_unique_indexes = closest_neighbors

    n = len(closest_neighbors)
    block_points = point_support.get_points_array(
        block_id=unknown_block_index
    )
    tot_unknown_value = np.sum(block_points[:, 2])

    sill = semivariogram_model.sill

    # Get block to point weighted semivariances
    b2p_semivariances = kriging_data.point_to_block_ps_semivariance_array()

    # Transform to covariances
    avg_b2p_covariances = sem_to_cov(b2p_semivariances, sill)

    # Add ones
    avg_b2p_covariances = add_ones(
        avg_b2p_covariances
    )

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

    # Solve Kriging systems
    # Each point of unknown area represents different kriging system

    # Transpose points matrix
    transposed = (np.nan_to_num(avg_b2p_covariances)).T

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

    predicted_points = []

    # todo: for loop into a function
    for idx, upoint in enumerate(block_points):
        vals = transposed[idx]

        try:
            w = np.linalg.solve(weights, vals)
        except TypeError:
            weights = weights.astype(float)
            vals = vals.astype(float)
            w = np.linalg.solve(weights, vals)

        zhat = block_values.dot(w[:-1])

        if zhat < 0:
            if raise_when_negative_prediction:
                if err_to_nan:
                    predicted_points.append(
                        [upoint[0], upoint[1], np.nan, np.nan]
                    )
                    continue
                else:
                    raise ValueError(f'Predicted value is {zhat} and it '
                                     f'should not be lower than 0. Check your '
                                     f'sampling grid, samples, number of '
                                     f'neighbors or semivariogram model type.')

        point_pop = upoint[2]
        zhat = (zhat * point_pop) / tot_unknown_value

        # Calculate error
        _sigmasq = np.matmul(w.T,
                             vals)
        sigmasq = covariance_within_unknown[0] - _sigmasq

        if sigmasq < 0:
            if raise_when_negative_error:
                if err_to_nan:
                    predicted_points.append(
                        [upoint[0], upoint[1], zhat, np.nan]
                    )
                    continue
                else:
                    raise ValueError(f'Predicted error value is {sigmasq} and '
                                     f'it should not be lower than 0. '
                                     f'Check your sampling grid, samples, '
                                     f'number of neighbors or semivariogram '
                                     f'model type.')
            else:
                sigma = 0
        else:
            sigma = np.sqrt(sigmasq)

        predicted_points.append([upoint[0], upoint[1], zhat, sigma])

    parr = np.asarray(predicted_points)
    return {unknown_block_index: parr}
