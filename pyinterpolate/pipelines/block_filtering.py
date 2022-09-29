"""
The Area-to-Area, Area-to-Point, Centroid-based Poisson Kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

TODO
----
* impute 0s in ata, atp and cb if val < 0
"""
from datetime import datetime
from typing import Union, Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from pyinterpolate.kriging.models.block.area_to_area_poisson_kriging import area_to_area_pk
from pyinterpolate.kriging.models.block.area_to_point_poisson_kriging import area_to_point_pk
from pyinterpolate.kriging.models.block.centroid_based_poisson_kriging import centroid_poisson_kriging
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, transform_blocks_to_numpy

from pyinterpolate.variogram import TheoreticalVariogram


class BlockPK:
    """
    Class is an object that can be used for Area-to-Area, Area-to-Point, Centroid-based Poisson Kriging
    regularization.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        The fitted variogram model.

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

    kriging_type : str, default='ata'
        A type of Poisson Kriging operation. Available methods:
          * ``'ata'``: Area-to-Area Poisson Kriging.
          * ``'atp'``: Area-to-Point Poisson Kriging.
          * ``'cb'``: Centroid-based Poisson Kriging.

    Attributes
    ----------
    semivariogram_model : TheoreticalVariogram
        See the ``semivariogram_model`` parameter.

    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
        See the ``blocks`` parameter.

    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
        See the ``point_support`` parameter.

    kriging_type : str, default='ata'
        See the ``kriging_type`` parameter.

    geo_ds : geopandas GeoDataFrame
        A regularized set of blocks: ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``

    statistics : Dict
        A dictionary with two keys:
          * ``'RMSE'``: root mean squared error of regularization,
          * ``'time'``: time (in seconds) of the regularization process.

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    Methods
    -------
    regularize()
        Regularize blocks (you should use it for a data deconvolution - with ATP PK, or for a data filtering - with
        ATA, C-B PK).
    """

    def __init__(self,
                 semivariogram_model: TheoreticalVariogram,
                 blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                 point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                 kriging_type='ata'):

        self.kriging_type = kriging_type
        # Check kriging type
        self._check_given_kriging_type()

        # Check if semivariogram exists
        self.semivariogram_model = semivariogram_model
        # Here IDE (PyCharm) gets type inspection wrong...
        # noinspection PyTypeChecker
        self.blocks = blocks
        # Here IDE (PyCharm) gets type inspection wrong...
        # noinspection PyTypeChecker
        self.point_support = point_support

        # Final GeoDataFrame with regularized data
        self.geo_ds = None

        # Error of regularization
        self.statistics = {}

        # Control
        self.raise_when_negative_prediction = True
        self.raise_when_negative_error = True

    @staticmethod
    def _rmse(prediction, real_value):
        # Pred row: idx, value, error
        rmse = np.sqrt((real_value - prediction) ** 2)

        return rmse

    def _interpolate(self, bl_arr, ps_dict, ids, regularized_area_id, n_neighbours, pred_raise, err_raise) -> List:

        # Get unknown blocks and ps
        u_block = bl_arr[bl_arr[:, 0] == regularized_area_id][:, :-1]
        u_point_support = ps_dict[regularized_area_id]

        # Get known blocks and ps
        k_ps = {}
        for _id in ids:
            if _id != regularized_area_id:
                k_ps[_id] = ps_dict[_id]

        k_bl_arr = bl_arr[bl_arr[:, 0] != regularized_area_id]

        # Model
        model_output = []
        if self.kriging_type == 'ata':
            model_output = area_to_area_pk(semivariogram_model=self.semivariogram_model,
                                           blocks=k_bl_arr,
                                           point_support=k_ps,
                                           unknown_block=u_block,
                                           unknown_block_point_support=u_point_support,
                                           number_of_neighbors=n_neighbours,
                                           raise_when_negative_prediction=pred_raise,
                                           raise_when_negative_error=err_raise)
        elif self.kriging_type == 'atp':
            model_output = area_to_point_pk(semivariogram_model=self.semivariogram_model,
                                            blocks=k_bl_arr,
                                            point_support=k_ps,
                                            unknown_block=u_block,
                                            unknown_block_point_support=u_point_support,
                                            number_of_neighbors=n_neighbours,
                                            raise_when_negative_prediction=pred_raise,
                                            raise_when_negative_error=err_raise)
        elif self.kriging_type == 'cb':
            model_output = centroid_poisson_kriging(semivariogram_model=self.semivariogram_model,
                                                    blocks=k_bl_arr,
                                                    point_support=k_ps,
                                                    unknown_block=u_block,
                                                    unknown_block_point_support=u_point_support,
                                                    number_of_neighbors=n_neighbours,
                                                    raise_when_negative_prediction=pred_raise,
                                                    raise_when_negative_error=err_raise)
        else:
            self._raise_wrong_kriging_type_error()

        return model_output

    def regularize(self,
                   number_of_neighbors,
                   data_crs=None,
                   raise_when_negative_prediction=True,
                   raise_when_negative_error=True):
        """
        Function regularizes whole dataset and creates new values and error maps based on the kriging type. Function
        does not predict unknown and missing values, areas with ``NaN`` values are skipped.

        Parameters
        ----------
        number_of_neighbors : int
            The minimum number of neighbours that potentially affect block.

        data_crs : str, default=None
            Data crs, look into: https://geopandas.org/projections.html. If None given then returned
            GeoDataFrame doesn't have a crs.

        raise_when_negative_prediction : bool, default=True
            Raise error when prediction is negative.

        raise_when_negative_error : bool, default=True
            Raise error when prediction error is negative.

        Returns
        -------
        regularized : gpd.GeoDataFrame
            Regularized set of blocks: ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``
        """
        t_start = datetime.now()
        # Transform point support to dict
        if isinstance(self.point_support, Dict):
            dict_ps = self.point_support
        else:
            dict_ps = transform_ps_to_dict(self.point_support)

        # Transform Blocks to array
        if isinstance(self.blocks, np.ndarray):
            arr_bl = self.blocks
        else:
            # Here IDE (PyCharm) gets type inspection wrong...
            # noinspection PyTypeChecker
            arr_bl = transform_blocks_to_numpy(self.blocks)

        # Get IDs - only those that are present in the blocks and point support!
        common_ids = set(dict_ps.keys()) & set(arr_bl[:, 0])

        # TODO: Warn if common ids are less than ids in Point Support or Blocks
        # if len(common_ids) != len(set(dict_ps.keys())):
        #     # TODO warning
        #     pass
        #
        # if len(common_ids) != len(set(dict_bl.keys())):
        #     # TODO warning
        #     pass

        list_of_interpolated_results = []

        for block_id in common_ids:
            prediction_rows = self._interpolate(bl_arr=arr_bl,
                                                ps_dict=dict_ps,
                                                ids=common_ids,
                                                regularized_area_id=block_id,
                                                n_neighbours=number_of_neighbors,
                                                pred_raise=raise_when_negative_prediction,
                                                err_raise=raise_when_negative_error)

            # Order interpolated values for GeoDataFrame
            results = self._parse_results(prediction_rows, arr_bl[arr_bl[:, 0] == block_id])
            for res in results:
                list_of_interpolated_results.append(res)

        # Transform results into a dataframe
        gdf = self._to_gdf(list_of_interpolated_results, data_crs)

        # Update stats
        t_end = (t_start - datetime.now()).seconds

        self._update_stats(gdf, t_end)
        self.geo_ds = gdf
        return gdf

    def _check_given_kriging_type(self):
        kriging_fns = ['ata', 'atp', 'cb']
        if self.kriging_type not in kriging_fns:
            self._raise_wrong_kriging_type_error()

    @staticmethod
    def _raise_wrong_kriging_type_error():
        l1 = 'Provided argument is not correct. You must choose kriging type.\n'
        l2 = "'ata' - area to area Poisson Kriging,\n"
        l3 = "'atp' - area to point Poisson Kriging,\n."
        l4 = "'cb' - centroid-based Poisson Kriging."
        message = l1 + l2 + l3 + l4
        raise KeyError(message)

    def _parse_results(self, prediction_rows: List, interpolated_block):
        """
        Function parses data to form: Regularized set of blocks: ['id', 'geometry', 'reg.est', 'reg.err', 'rmse'].

        Parameters
        ----------
        prediction_rows

        Returns
        -------
        parsed_results : List
                         [index, geometry, prediction, error]
        """
        if interpolated_block.shape[0] == 1:
            interpolated_block = interpolated_block[0]

        if self.kriging_type == 'ata':
            parsed_results = [[prediction_rows[0],
                              Point([interpolated_block[1], interpolated_block[2]]),
                              prediction_rows[1],
                              prediction_rows[2],
                              self._rmse(prediction_rows[1], interpolated_block[-1])]]
            return parsed_results
        elif self.kriging_type == 'atp':
            predicted_sum = np.sum(np.array([x[1] for x in prediction_rows]))
            real_value = interpolated_block[-1]
            rmse = self._rmse(predicted_sum, real_value)
            parsed_results = []

            for row in prediction_rows:
                parsed = [interpolated_block[0],
                          Point(row[0]),
                          row[1],
                          row[2],
                          rmse]
                parsed_results.append(parsed)
            return parsed_results

        elif self.kriging_type == 'cb':
            parsed_results = [[prediction_rows[0],
                              Point([interpolated_block[1], interpolated_block[2]]),
                              prediction_rows[1],
                              prediction_rows[2],
                              self._rmse(prediction_rows[1], interpolated_block[-1])]]
            return parsed_results
        else:
            self._raise_wrong_kriging_type_error()

    @staticmethod
    def _to_gdf(list_of_results, crs):
        gdf = gpd.GeoDataFrame(list_of_results)
        gdf.columns = ['id', 'geometry', 'reg.est', 'reg.err', 'rmse']
        gdf.geometry = gdf['geometry']
        if crs is not None:
            gdf.crs = crs
        return gdf

    def _update_stats(self, gdf, t_end):
        self.statistics['time (s)'] = t_end
        self.statistics['RMSE'] = np.mean(gdf['rmse'])
