from typing import Dict, List
import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.kriging.models.block.weight import weights_array
from pyinterpolate.processing.select_values import select_centroid_poisson_kriging_data
from pyinterpolate.variogram import TheoreticalVariogram


def centroid_poisson_kriging(semivariogram_model: TheoreticalVariogram,
                             blocks: Dict,
                             point_support: Dict,
                             unknown_block: np.ndarray,
                             unknown_block_point_support: np.ndarray,
                             number_of_neighbors: int,
                             max_neighbors_radius: float,
                             is_weighted_by_point_support=True,
                             raise_when_anomalies=False) -> List:
    """
    Function performs centroid-based Poisson Kriging of blocks (areal) data.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
                          Fitted variogram.

    blocks : Dict
             Dictionary retrieved from the Blocks, it's structure is defined as:
             polyset = {
                      'geometry': {
                          'block index': geometry
                      }
                      'data': [[index centroid.x, centroid.y value]],
                      'info': {
                          'index_name': the name of the index column,
                          'geometry_name': the name of the geometry column,
                          'value_name': the name of the value column,
                          'crs': CRS of a dataset
                      }
                  }

    point_support : Dict
                    Point support data as a Dict:

                        point_support = {
                            'area_id': [numpy array with points]
                        }

    unknown_block : numpy array
                    [index, centroid.x, centroid.y]

    unknown_block_point_support : numpy array
                                  Points within block [[x, y, point support value]]

    number_of_neighbors : int
                          The minimum number of neighbours that potentially affect block.

    max_neighbors_radius : float
                           The maximum radius of search for the closest neighbors.

    is_weighted_by_point_support : bool, default = True
                                   Are distances between blocks weighted by point support?

    raise_when_anomalies : bool, default = False
                           Raise ValueError if kriging weights are negative.

    Returns
    -------
    results : List
              [unknown block index, prediction, error]

    """
    # Get data: [block id, cx, cy, value, distance to unknown, aggregated point support sum]
    kriging_data = select_centroid_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_blocks=blocks,
        k_point_support=point_support,
        nn=number_of_neighbors,
        max_radius=max_neighbors_radius,
        weighted=is_weighted_by_point_support
    )

    n = len(kriging_data)
    distances = kriging_data[:, 4]
    values = kriging_data[:, 3]

    partial_semivars = semivariogram_model.predict(distances)
    semivars = np.ones(len(partial_semivars) + 1)
    semivars[:-1] = partial_semivars
    semivars = semivars.transpose()

    # Distances between known blocks
    coordinates = kriging_data[:, 1:3]
    block_distances = calc_point_to_point_distance(coordinates).flatten()
    known_blocks_semivars = semivariogram_model.predict(block_distances)
    predicted = np.array(known_blocks_semivars.reshape(n, n))

    # Add diagonal weights to predicted semivars array
    weights = weights_array(predicted.shape, values, kriging_data[:, 5])
    weighted_and_predicted = predicted + weights

    # Prepare matrix for solving kriging system
    ones_col = np.ones((weighted_and_predicted.shape[0], 1))
    weighted_and_predicted = np.c_[weighted_and_predicted, ones_col]
    ones_row = np.ones((1, weighted_and_predicted.shape[1]))
    ones_row[0][-1] = 0
    kriging_weights = np.r_[weighted_and_predicted, ones_row]

    # Solve Kriging system
    try:
        w = np.linalg.solve(kriging_weights, semivars)
    except TypeError:
        kriging_weights = kriging_weights.astype(float)
        semivars = semivars.astype(float)
        w = np.linalg.solve(kriging_weights, semivars)

    zhat = values.dot(w[:-1])

    sigmasq = (w.T * semivars)[0]

    if sigmasq < 0:
        if raise_when_anomalies:
            msg = f'Predicted variance is below 0 == {sigmasq}. Check your dataset for clustered data or change ' \
                  f'the variogram model type'
            raise ValueError(msg)
        sigma = 0
    else:
        sigma = np.sqrt(sigmasq)

    # Prepare output
    results = [unknown_block[0], zhat, sigma]
    return results
