from typing import Union, Dict

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.kriging.models.block.weight import WeightedBlock2BlockSemivariance, WeightedBlock2PointSemivariance, \
    add_ones
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.select_values import select_poisson_kriging_data
from pyinterpolate.processing.transform.transform import transform_ps_to_dict
from pyinterpolate.variogram import TheoreticalVariogram


def area_to_point_pk(semivariogram_model: TheoreticalVariogram,
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
    # Get total point-support value of the unknown area
    tot_unknown_value = np.sum(unknown_block_point_support[:, -1])

    # Transform point support to dict
    if isinstance(point_support, Dict):
        dps = point_support
    else:
        dps = transform_ps_to_dict(point_support)

    # Prepare Kriging Data
    # {known block id: [unknown_pt_idx_coordinates, known pt val, unknown pt val, distance between points]}
    kriging_data = select_poisson_kriging_data(
        u_block_centroid=unknown_block,
        u_point_support=unknown_block_point_support,
        k_point_support_dict=dps,
        nn=number_of_neighbors)

    # Get block to block weighted semivariances
    b2b_semivariance = WeightedBlock2BlockSemivariance(semivariance_model=semivariogram_model)
    # {known area_id: weighted block-to-block semivariance between unknown area and known area}
    avg_b2b_semivariances = b2b_semivariance.calculate_average_semivariance(kriging_data)

    # Get block to point weighted semivariances
    b2p_semivariance = WeightedBlock2PointSemivariance(semivariance_model=semivariogram_model)
    # array(
    #     [unknown point 1 (n) semivariance against point support from block 1,
    #      unknown point 2 (n+1) semivariance against point support from block 1,
    #      ...],
    #     [unknown point 1 (n) semivariance against point support from block 2,
    #      unknown point 2 (n+1) semivariance against point support from block 2,
    #      ...],
    # )
    avg_b2p_semivariances = add_ones(b2p_semivariance.calculate_average_semivariance(kriging_data))



    return avg_b2b_semivariances, avg_b2p_semivariances




