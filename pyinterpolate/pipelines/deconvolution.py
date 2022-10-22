"""
Regularization: blocks to point support.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

TODO
----
* tests
"""
from typing import Union, Dict, Any

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Point
from tqdm import tqdm

from pyinterpolate.kriging.models.block import area_to_point_pk
from pyinterpolate.processing.checks import check_ids
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, transform_blocks_to_numpy
from pyinterpolate.variogram import TheoreticalVariogram


def smooth_area_to_point_pk(semivariogram_model: TheoreticalVariogram,
                            blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                            point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                            number_of_neighbors: int,
                            max_range=None,
                            crs: Any = None,
                            raise_when_negative_prediction=True,
                            raise_when_negative_error=True,
                            err_to_nan=True) -> gpd.GeoDataFrame:
    """
    Function smooths blocks data into their point support values.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        The regularized variogram.

    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
        Blocks with aggregated data.
            * ``Blocks``: ``Blocks()`` class object.
            * ``GeoDataFrame`` and ``DataFrame`` must have columns: ``centroid.x, centroid.y, ds, index``.
              Geometry column with polygons is not used.
            * ``numpy array``: ``[[block index, centroid x, centroid y, value]]``.

    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
        The point support of polygons.
          * ``Dict``: ``{block id: [[point x, point y, value]]}``,
          * ``numpy array``: ``[[block id, x, y, value]]``,
          * ``DataFrame`` and ``GeoDataFrame``: ``columns={x, y, ds, index}``,
          * ``PointSupport``.

    number_of_neighbors : int
        The minimum number of neighbours that potentially affect block.

    max_range : float, default=None
        The maximum distance to search for neighbors.

    crs : Any, default=None
        CRS of data.

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    err_to_nan : bool, default=True
        ``ValueError`` to ``NaN``.


    Returns
    -------
    results : gpd.GeoDataFrame
        Columns = ``[area_id, geometry (Point), prediction, error]``.

    TODO
    ----
    - Check strange parameter number_of_neighbors (not min but max?)
    """

    # Prepare data
    # Transform point support to dict
    if isinstance(point_support, Dict):
        dict_ps = point_support.copy()
    else:
        dict_ps = transform_ps_to_dict(point_support)

    # Transform Blocks to array
    if isinstance(blocks, np.ndarray):
        arr_bl = blocks
    else:
        # Here IDE (PyCharm) gets type inspection wrong...
        # noinspection PyTypeChecker
        arr_bl = transform_blocks_to_numpy(blocks)

    rarr = []
    block_ids = arr_bl[:, 0]
    ps_ids = list(dict_ps.keys())
    possible_idx = set(block_ids) & set(ps_ids)

    possible_len = len(possible_idx)

    if possible_len != len(block_ids) or possible_len != len(ps_ids):
        check_ids(block_ids, ps_ids, set_name_a='Blocks indexes', set_name_b='Point Support indexes')

    for area_id in tqdm(arr_bl[:, 0]):
        if area_id in possible_idx:
            k_areas = arr_bl[arr_bl[:, 0] != area_id].copy()
            exclude_key = {area_id}
            k_points = {k: dict_ps[k] for k in set(list(dict_ps.keys())) - exclude_key}
            u_area = arr_bl[arr_bl[:, 0] == area_id].copy()
            u_points = dict_ps[area_id].copy()

            results = area_to_point_pk(semivariogram_model=semivariogram_model,
                                       blocks=k_areas,
                                       point_support=k_points,
                                       unknown_block=u_area[0][:-1],
                                       unknown_block_point_support=u_points,
                                       number_of_neighbors=number_of_neighbors,
                                       max_range=max_range,
                                       raise_when_negative_prediction=raise_when_negative_prediction,
                                       raise_when_negative_error=raise_when_negative_error,
                                       err_to_nan=err_to_nan)

            for result in results:
                pred_arr = [area_id, Point(result[0]), result[1], result[2]]
                rarr.append(pred_arr)

    gdf = gpd.GeoDataFrame(data=rarr, columns=['area id', 'geometry', 'pred', 'err'])
    gdf.geometry = gdf['geometry']

    if crs is not None:
        gdf.set_crs(crs=crs, inplace=True)

    return gdf
