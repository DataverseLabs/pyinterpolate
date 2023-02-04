"""
Array weighting classes and functions designed to solve kriging system for block Poisson Kriging models.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Tuple, Dict
import numpy as np
from pyinterpolate.variogram import TheoreticalVariogram


class WeightedBlock2BlockSemivariance:
    """
    Class calculates the average weighted block-to-block semivariance.

    Parameters
    ----------
    semivariance_model : TheoreticalVariogram
                         Fitted variogram model.

    Attributes
    ----------
    semivariance_model : TheoreticalVariogram
                         See semivariance_model parameter.

    Methods
    -------
    calculate_average_semivariance(datarows: Dict)
        Function calculates the average semivariance from a given set of points and distances between them.

    Notes
    -----

    Weighted semivariance is calculated as:

    (1)

    $$\gamma_{v_{i}, v_{j}}
        =
        \frac{1}{\sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'}} *
            \sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'} * \gamma(u_{s}, u_{s'})$$

    where:
    * $w_{ss'}$ - product of point-support weights from block a and block b.
    * $\gamma(u_{s}, u_{s'})$ - semivariance between point-supports of block a and block b.

    Examples
    --------
    avg_semivar = WeightedBlock2BlockSemivariance(variogram_model)
    semivars = avg_semivar.calculate_average_semivariance(blocks_data)
    print(semivars)
    >>> {'block x': 200, 'block y': 300}
    """

    def __init__(self, semivariance_model):
        self.semivariance_model = semivariance_model

    def _avg_smv(self, datarows: np.ndarray) -> Tuple:
        """
        Function calculates weight and partial semivariance of a data row [point value a, point value b, distance]

        Parameters
        ----------
        datarows : numpy array
                   [point value a, point value b, distance between points]

        Returns
        -------
        : Tuple[float, float]
            [Weighted sum of semivariances, Weights sum]
        """
        if datarows.ndim == 1:
            datarows = datarows[np.newaxis, :]

        predictions = self.semivariance_model.predict(datarows[:, -1])

        weights = datarows[:, 0] * datarows[:, 1]
        summed_weights = np.sum(weights)
        summed_semivariances = np.sum(
            predictions * weights
        )

        return summed_semivariances, summed_weights

    def calculate_average_semivariance(self, data_points: Dict) -> Dict:
        """
        Function calculates the average semivariance.

        Parameters
        ----------
        data_points : Dict
                      {known block id: [(unknown x, unknown y), [unknown val, known val, distance between points]]}

        Returns
        -------
        weighted_semivariances : Dict
                                 {area_id: weighted semivariance}
        """
        k = {}
        for idx, prediction_input in data_points.items():

            _input = prediction_input

            if isinstance(prediction_input[0][0], Tuple):
                _input = prediction_input[1]

            if len(prediction_input) == 2:
                if isinstance(prediction_input[0], np.ndarray) and isinstance(prediction_input[1], np.ndarray):
                    _input = prediction_input[1]


            w_sem = self._avg_smv(_input)
            w_sem_sum = w_sem[0]
            w_sem_weights_sum = w_sem[1]

            k[idx] = w_sem_sum / w_sem_weights_sum

        return k


class WeightedBlock2PointSemivariance:
    """
    Class calculates the average weighted block-to-point semivariance.

    Parameters
    ----------
    semivariance_model : TheoreticalVariogram
                         Fitted variogram model.

    Attributes
    ----------
    semivariance_model : TheoreticalVariogram
                         See semivariance_model parameter.

    Methods
    -------
    calculate_average_semivariance(datarows: Dict)
        Function calculates the average semivariance from a given set of points and distances between them.

    Notes
    -----
    Weighted semivariance is calculated as:

        (1)

    $$\gamma_{v_{i}, u_{s}}
        =
        \frac{1}{\sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'}} *
            \sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'} * \gamma(u_{s}, u_{s'})$$

    where:
        * $w_{ss'}$ - product of point-support weights from block a and block b.
        * $\gamma(u_{s}, u_{s'})$ - semivariance between point-supports of block a and block b.
        * $P_{j}=1$ - only one semivariance value between a single point and all other points at a time.

    Examples
    --------
    avg_semivar = WeightedBlock2PointSemivariance(variogram_model)
    semivars = avg_semivar.calculate_average_semivariance(blocks_data)
    print(semivars)
    >>> ([[100, 200], [300, 200], [200, 100]])
    """

    def __init__(self, semivariance_model: TheoreticalVariogram):
        self.semivariance_model = semivariance_model

    def _avg_smv(self, unknown_pt_value, known_points_values_and_distances):
        """
        Function calculates weight and partial semivariance of a data row [point value a, point value b, distance]

        Parameters
        ----------
        unknown_pt_value : float
                           The value of an unknown point.

        known_points_values_and_distances : numpy array
                                            [known pt val, distance between points]
        Returns
        -------
        pt_semi : float
                  A weighted semivariance.
        """

        if unknown_pt_value > 0:
            partial_semivars = self.semivariance_model.predict(known_points_values_and_distances[:, 1])

            all_weights = unknown_pt_value * known_points_values_and_distances[:, 0]

            weighted_semivars = np.sum(partial_semivars * all_weights)

            all_weights_sum = np.sum(all_weights)
            if all_weights_sum == 0:
                weighted_block_smv = 0
            else:
                weighted_block_smv = weighted_semivars / np.sum(all_weights)
        else:
            weighted_block_smv = 0

        return weighted_block_smv

    def calculate_average_semivariance(self, data_points: Dict) -> np.ndarray:
        """
        Function calculates average semivariance a single point from block Pj and all points from block Pi.

        Parameters
        ----------
        data_points : Dict
                      {
                          known block id:
                              [(unknown x, unknown y), array(unknown val, known val, distance between points)]
                      }
                      or
                      {
                          block Pi id:
                              [
                                  (block Pj point support point x_n, block Pj point support point y_n),
                                  array(
                                      [value of unknown block (Pj) point-support point n,
                                      value of known block (Pi) point-support point i,
                                      distance between (x_n, y_n) and (x_i, y_i)],
                                      [value of unknown block (Pj) point-support point n,
                                      value of known block (Pi) point-support point i+1,
                                      distance between (x_n, y_n) and (x_i+1, y_i+1)],
                                      [...]
                                      )
                              ]
                      }


        Returns
        -------
        point_to_block_semivariances : numpy array
                                       Predicted and weighted semivariances for each unknown block point support
                                       against known blocks.
                                       array(
                                            [unknown point 1 (n) semivariance against point support from block 1,
                                            unknown point 2 (n+1) semivariance against point support from block 1,
                                            ...],
                                            [unknown point 1 (n) semivariance against point support from block 2,
                                            unknown point 2 (n+1) semivariance against point support from block 2,
                                            ...],
                                       )
        """

        point_to_blocks_smvs = []
        for area_index, data in data_points.items():
            pts_per_area = []
            block_values_and_distances = data[1]
            points = set(data[0])
            for single_unknown_point in points:
                mask = [x == single_unknown_point for x in data[0]]
                known_points_and_distances = block_values_and_distances[mask]
                unknown_point_value = block_values_and_distances[0][0]
                other_points = known_points_and_distances[:, 1:]
                pt_output = self._avg_smv(unknown_point_value, other_points)
                pts_per_area.append(pt_output)
            point_to_blocks_smvs.append(pts_per_area)
        return np.array(point_to_blocks_smvs)


def add_ones(array: np.ndarray) -> np.ndarray:
    """Function adds rows of ones to a given array.

    Parameters
    ----------
    array : numpy array
            Array of size MxN (M rows, N cols)
    Returns
    -------
    list_with_ones : numpy array
                     Array of size M+1xN (M+1 rows, N cols) where the last row are N ones.
    """
    ones = np.ones(np.shape(array)[1])
    list_with_ones = np.vstack((array, ones))
    return list_with_ones


def weights_array(predicted_semivariances_shape, block_vals, point_support_vals) -> np.array:
    """
    Function calculates additional diagonal weights for the matrix of predicted semivariances.

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
