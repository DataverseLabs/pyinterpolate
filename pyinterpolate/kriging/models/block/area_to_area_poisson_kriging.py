from typing import Dict, Union, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.kriging.models.block.weight import weights_array
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_ata_poisson_kriging_data, prepare_ata_pk_known_areas,\
    get_aggregated_point_support_values, get_distances_within_unknown
from pyinterpolate.processing.transform.transform import get_areal_values_from_agg
from pyinterpolate.variogram import TheoreticalVariogram


class WeightedBlock2BlockSemivariance:

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
                      {area id: numpy array with [pt a value, pt b value, distance between a and b]}

        Returns
        -------
        weighted_semivariances : Dict
                                 {area_id: weighted semivariance}

        Notes
        -----

        Weighted semivariance is calculated as:

        (1)

        $$\gamma_{w}=\frac{1}{\sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'}} * \sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'} * \gamma(u_{s}, u_{s'})$$

        where:
        * $w_{ss'}$ - product of point-support weights from block a and block b.
        * $\gamma(u_{s}, u_{s'})$ - semivariance between point-supports of block a and block b.
        """
        k = {}
        for idx, prediction_input in data_points.items():
            w_sem = self._avg_smv(prediction_input)
            w_sem_sum = w_sem[0]
            w_sem_weights_sum = w_sem[1]

            k[idx] = w_sem_sum / w_sem_weights_sum

        return k


def area_to_area_pk(semivariogram_model: TheoreticalVariogram,
                    blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                    point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                    unknown_block: np.ndarray,
                    unknown_block_point_support: np.ndarray,
                    number_of_neighbors: int):
    """
    Function predicts areal value in a unknown location based on the area-to-area Poisson Kriging

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
                          Regularized variogram.

    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
             Blocks with aggregated data.
             * Blocks: Blocks() class object.
             * GeoDataFrame and DataFrame must have columns: centroid.x, centroid.y, ds, index.
               Geometry column with polygons is not used and optional.
             * numpy array: [[block index, centroid x, centroid y, value]].

    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
                    * Dict: {block id: [[point x, point y, value]]}
                    * numpy array: [[block id, x, y, value]]
                    * DataFrame and GeoDataFrame: columns={x, y, ds, index}
                    * PointSupport

    unknown_block : numpy array
                    [index, centroid.x, centroid.y]

    unknown_block_point_support : numpy array
                                  Points within block [[x, y, point support value]]

    number_of_neighbors : int
                          The minimum number of neighbours that potentially affect block.


    Returns
    -------
    results : List
              [unknown block index, prediction, error]

    """
    # Prepare Kriging Data
    # {known block id: [known pt val, unknown pt val, distance between points]}

    kriging_data = select_ata_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_point_support=point_support,
        nn=number_of_neighbors)

    b2b_semivariance = WeightedBlock2BlockSemivariance(semivariance_model=semivariogram_model)
    avg_semivariances = b2b_semivariance.calculate_average_semivariance(kriging_data)

    k = np.array(list(avg_semivariances.values()))
    k = k.T
    k_ones = np.ones(len(k)+1)
    k_ones[:-1] = k

    # Prepare blocks for calculation
    prepared_ids = list(avg_semivariances.keys())

    # {(known block id a, known block id b): [pt a val, pt b val, distance between points]}
    distances_between_known_areas = prepare_ata_pk_known_areas(point_support, prepared_ids)

    semivariances_between_known_areas = b2b_semivariance.calculate_average_semivariance(
        distances_between_known_areas)

    # Create array
    predicted = []
    for idx_a in prepared_ids:
        row = []
        for idx_b in prepared_ids:
            row.append(semivariances_between_known_areas[(idx_a, idx_b)])
        predicted.append(row)

    predicted = np.array(predicted)

    # Add diagonal weights
    values = get_areal_values_from_agg(blocks, prepared_ids)
    aggregated_ps = get_aggregated_point_support_values(point_support, prepared_ids)
    weights = weights_array(predicted.shape, values, aggregated_ps)
    weighted_and_predicted = predicted + weights

    # Prepare weights matrix
    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[weighted_and_predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    try:
        w = np.linalg.solve(weights, k_ones)
    except TypeError:
        weights = weights.astype(np.float)
        k_ones = k_ones.astype(np.float)
        w = np.linalg.solve(weights, k_ones)

    zhat = values.dot(w[:-1])

    # Calculate prediction error

    if isinstance(unknown_block[0], np.ndarray):
        u_idx = unknown_block[0][0]
    else:
        u_idx = unknown_block[0]

    distances_within_unknown_block = get_distances_within_unknown(unknown_block_point_support)
    semivariance_within_unknown = b2b_semivariance.calculate_average_semivariance({
        u_idx: distances_within_unknown_block
    })[u_idx]

    sig_base = (w.T * k_ones)[0]
    sigmasq = semivariance_within_unknown - sig_base
    if sigmasq < 0:
        sigma = 0
    else:
        sigma = np.sqrt(sigmasq)
    return u_idx, zhat, sigma
