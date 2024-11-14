from typing import Tuple

import numpy as np


# class WeightedBlock2BlockSemivariance:
#     """
#     Class calculates the average weighted block-to-block semivariance.
#
#     Parameters
#     ----------
#     theoretical_model : TheoreticalVariogram
#                          Fitted variogram model.
#
#     Attributes
#     ----------
#     theoretical_model : TheoreticalVariogram
#                          See semivariance_model parameter.
#
#     Methods
#     -------
#     calculate_average_semivariance(datarows: Dict)
#         Function calculates the average semivariance from a given set of points and distances between them.
#
#     Notes
#     -----
#
#     Weighted semivariance is calculated as:
#
#     (1)
#
#     $$\gamma_{v_{i}, v_{j}}
#         =
#         \frac{1}{\sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'}} *
#             \sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'} * \gamma(u_{s}, u_{s'})$$
#
#     where:
#     * $w_{ss'}$ - product of point-support custom_weights from block a and block b.
#     * $\gamma(u_{s}, u_{s'})$ - semivariance between point-supports of block a and block b.
#
#     Examples
#     --------
#     avg_semivar = WeightedBlock2BlockSemivariance(variogram_model)
#     semivars = avg_semivar.calculate_average_semivariance(blocks_data)
#     print(semivars)
#     >>> {'block x': 200, 'block y': 300}
#     """
#
#     def __init__(self, theoretical_model):
#         self.theoretical_model = theoretical_model
#
#     def _avg_smv(self, datarows: np.ndarray) -> Tuple:
#         """
#         Function calculates weight and partial semivariance of a data row [point value a, point value b, distance]
#
#         Parameters
#         ----------
#         datarows : numpy array
#                    [point value a, point value b, distance between points]
#
#         Returns
#         -------
#         : Tuple[float, float]
#             [Weighted sum of semivariances, Weights sum]
#         """
#         if datarows.ndim == 1:
#             datarows = datarows[np.newaxis, :]
#
#         predictions = self.theoretical_model.predict(datarows[:, -1])
#
#         custom_weights = datarows[:, 0] * datarows[:, 1]
#         summed_weights = np.sum(custom_weights)
#         summed_semivariances = np.sum(
#             predictions * custom_weights
#         )
#
#         return summed_semivariances, summed_weights
#
#     def calculate_average_semivariance(self, data_points: Dict) -> Dict:
#         """
#         Function calculates the average semivariance.
#
#         Parameters
#         ----------
#         data_points : Dict
#                       {known block id: [(unknown x, unknown y), [unknown val, known val, distance between points]]}
#
#         Returns
#         -------
#         weighted_semivariances : Dict
#                                  {area_id: weighted semivariance}
#         """
#         k = {}
#         for idx, prediction_input in data_points.items():
#
#             _input = prediction_input
#
#             if isinstance(prediction_input[0][0], Tuple):
#                 _input = prediction_input[1]
#
#             if len(prediction_input) == 2:
#                 if isinstance(prediction_input[0], np.ndarray) and isinstance(prediction_input[1], np.ndarray):
#                     _input = prediction_input[1]
#
#
#             w_sem = self._avg_smv(_input)
#             w_sem_sum = w_sem[0]
#             w_sem_weights_sum = w_sem[1]
#
#             k[idx] = w_sem_sum / w_sem_weights_sum
#
#         return k


def _weights_array(predicted_semivariances_shape, block_vals, point_support_vals) -> np.array:
    """
    Function calculates additional diagonal custom_weights for the matrix of predicted semivariances.

    Parameters
    ----------
    predicted_semivariances_shape : Tuple
        The size of semivariances array (nrows x ncols).

    block_vals : numpy array
        Array with values to calculate diagonal weight.

    point_support_vals : numpy array
        Array with values to calculate diagonal weight.

    Returns
    -------
    : numpy array
        The mask with zeros and diagonal weight of size (nrows x ncols).
    """

    weighted_array = np.sum(block_vals * point_support_vals)
    weight = weighted_array / np.sum(point_support_vals)
    w = np.zeros(shape=predicted_semivariances_shape)

    np.fill_diagonal(w, weight)
    return w
