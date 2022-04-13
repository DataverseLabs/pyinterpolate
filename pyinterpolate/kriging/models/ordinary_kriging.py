# Python core
from typing import Union, Collection, List

# Core calculation and data visualization
import numpy as np

# Pyinterpolate
from pyinterpolate.variogram import TheoreticalVariogram


def ordinary_kriging(
        theoretical_variogram_model: TheoreticalVariogram,
        unknown_locations: Union[np.ndarray, Collection],
        neighbors_range=None,
        min_no_neighbors=1,
        max_no_neighbors=-1,
        monitor_negative_weights=True
) -> List:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_variogram_model : TheoreticalVariogram
                                  Trained theoretical variogram model.

    unknown_locations : numpy array
                        Array with unknown locations.

    neighbors_range : float
                      Maximum distance where we search for point neighbors.

    min_no_neighbors : int
                       Minimum number of neighbors to estimate unknown value; value is used when insufficient number of
                       neighbors is within neighbors_range.

    max_no_neighbors : int
                       Maximum number of n-closest neighbors used for interpolation if there are too many neighbors
                       in neighbors_range. It speeds up calculations for large datasets.

    monitor_negative_weights : bool
                               Analyze output and weights and if they are negative throw warning.

    Returns
    -------
    : numpy array
        [longitude (x), latitude (y), predicted value, variance error]

    Warns
    -----
    NegativeWeightsWarning : set if weights in weighting matrix are negative.

    """

    # Check if variogram model is valid
    # TODO

    # Check range
    if neighbors_range is None:
        neighbors_range = theoretical_variogram_model.rang
        
    output = []

    for pt in unknown_locations:
        prepared_data = prepare_kriging_data(unknown_position=unknown_location,
                                             data_array=self.dataset,
                                             neighbors_range=neighbors_range,
                                             min_number_of_neighbors=min_no_neighbors,
                                             max_number_of_neighbors=max_no_neighbors)
        n = len(prepared_data)
        unknown_distances = prepared_data[:, -1]
        k = self.model.predict(unknown_distances)
        k = k.T
        k_ones = np.ones(1)[0]
        k = np.r_[k, k_ones]

        dists = calc_point_to_point_distance(prepared_data[:, :-2])

        predicted_weights = self.model.predict(dists.ravel())
        predicted = np.array(predicted_weights.reshape(n, n))
        p_ones = np.ones((predicted.shape[0], 1))
        predicted_with_ones_col = np.c_[predicted, p_ones]
        p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
        p_ones_row[0][-1] = 0.
        weights = np.r_[predicted_with_ones_col, p_ones_row]

        w = np.linalg.solve(weights, k)
        zhat = prepared_data[:, -2].dot(w[:-1])

        # Test for anomalies
        if test_anomalies:
            if zhat < 0:
                user_input_message = 'Estimated value is below zero and it is: {}. \n'.format(zhat)
                text_error = user_input_message + 'Program is terminated. Try different semivariogram model. ' \
                                                  '(Did you use gaussian model? \
                            If so then try to use other models like linear or exponential) and/or analyze your data \
                            for any clusters which may affect the final estimation'

                raise ValueError(text_error)

        sigma = np.matmul(w.T, k)
    return [zhat, sigma, w[-1], w]