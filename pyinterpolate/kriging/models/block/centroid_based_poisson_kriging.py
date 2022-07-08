from typing import Dict
import geopandas as gpd
import numpy as np

from pyinterpolate.processing.transform import prepare_poisson_kriging_data
from pyinterpolate.variogram import TheoreticalVariogram


def centroid_poisson_kriging(semivariogram_model: TheoreticalVariogram,
                             blocks: Dict,
                             point_support: Dict,
                             unknown_block: Dict,
                             unknown_block_point_support: Dict,
                             number_of_neighbors: int,
                             max_neighbors_radius: float,
                             is_weighted_by_point_support = True,
                             raise_when_anomalies = False) -> np.ndarray:
    """
    Function performs centroid-based Poisson Kriging of blocks (areal) data.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
                          Fitted variogram.

    blocks : Dict
             Dictionary retrieved from the PolygonDataClass, it's structure is defined as:

             polyset = {
                'blocks': {
                    'block index': {
                        'value_name': float,
                        'geometry_name': MultiPolygon | Polygon,
                        'centroid.x': float,
                        'centroid.y': float
                    }
                }
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

    unknown_block : Dict
                    'block index': {
                        'value_name': float,
                        'geometry_name': MultiPolygon | Polygon,
                        'centroid.x': float,
                        'centroid.y': float
                    }

    unknown_block_point_support : Dict
                                  {'block index': [numpy array with points]}

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
    results : numpy array
              [prediction, error, unknown block index]

    """

    kriging_data = prepare_poisson_kriging_data(
        u_block=unknown_block,
        u_point_support=unknown_block_point_support,
        k_blocks=blocks,
        k_point_support=point_support,
        nn=number_of_neighbors,
        max_radius=max_neighbors_radius,
        weighted=is_weighted_by_point_support
    )