from typing import Dict

import numpy as np

from pyinterpolate.variogram import TheoreticalVariogram


def area_to_area_pk(semivariogram_model: TheoreticalVariogram,
                    blocks: Dict,
                    point_support: Dict,
                    unknown_block: np.ndarray,
                    unknown_block_point_support: np.ndarray,
                    number_of_neighbors: int,
                    max_neighbors_radius: float):
    """
    Function predicts areal value in a unknown location based on the area-to-area Poisson Kriging

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
                          Regularized variogram.

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


    Returns
    -------
    results : List
              [unknown block index, prediction, error]

    """
    # Get data: [array[block id, block value],
    #            array[[know point support val, unknown point support val, distance between points], ...]]]




    pass
