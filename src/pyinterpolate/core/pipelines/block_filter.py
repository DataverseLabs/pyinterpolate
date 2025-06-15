"""
The Area-to-Area, Area-to-Point, Centroid-based Poisson Kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

TODO
----
* impute 0s in ata, atp and cb if val < 0?
"""
import time
from typing import Dict, List

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.point_support_distances import PointSupportDistance
from pyinterpolate.evaluate.metrics import root_mean_squared_error
from pyinterpolate.kriging.block.area_to_area_poisson_kriging import area_to_area_pk
from pyinterpolate.kriging.block.area_to_point_poisson_kriging import area_to_point_pk
from pyinterpolate.kriging.block.centroid_based_poisson_kriging import centroid_poisson_kriging
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def filter_blocks(semivariogram_model: TheoreticalVariogram,
                  point_support: PointSupport,
                  number_of_neighbors,
                  kriging_type='ata',
                  data_crs=None,
                  raise_when_negative_prediction=True,
                  raise_when_negative_error=False,
                  verbose=True) -> gpd.GeoDataFrame:
    """
    Function filters block data using Poisson Kriging. By filtering we
    understand computing aggregated values again using point support data
    for ratios regularization.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        The fitted variogram model.

    point_support : PointSupport
        Blocks and their point supports.

    number_of_neighbors : int
        The minimum number of neighbours that potentially affect a block.

    kriging_type : str, default='ata'
        A type of Poisson Kriging operation. Available methods:
          * ``'ata'``: Area-to-Area Poisson Kriging.
          * ``'atp'``: Area-to-Point Poisson Kriging.
          * ``'cb'``: Centroid-based Poisson Kriging.

    data_crs : str, default=None
        Data crs, look into: https://geopandas.org/projections.html.
        If None given then returned GeoDataFrame doesn't have a crs.

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    verbose : bool, default=True
        Show progress bar

    Returns
    -------
    : GeoPandas GeoDataFrame
        Regularized set of ps_blocks:
        ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``
    """

    block_k = BlockPoissonKriging(
        semivariogram_model=semivariogram_model,
        point_support=point_support,
        kriging_type=kriging_type,
        verbose=verbose
    )

    parsed = block_k.regularize(
        number_of_neighbors=number_of_neighbors,
        data_crs=data_crs,
        raise_when_negative_prediction=raise_when_negative_prediction,
        raise_when_negative_error=raise_when_negative_error
    )
    return parsed


def smooth_blocks(semivariogram_model: TheoreticalVariogram,
                  point_support: PointSupport,
                  number_of_neighbors,
                  data_crs=None,
                  raise_when_negative_prediction=True,
                  raise_when_negative_error=True,
                  verbose=True) -> gpd.GeoDataFrame:
    """
    Function smooths aggregated block values, and transform those into
    point support.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        The fitted variogram model.

    point_support : PointSupport
        Blocks and their point supports.

    number_of_neighbors : int
        The minimum number of neighbours that potentially affect a block.

    data_crs : str, default=None
        Data crs, look into: https://geopandas.org/projections.html.
        If ``None`` given then returned GeoDataFrame doesn't have a crs.

    raise_when_negative_prediction : bool, default=True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    verbose : bool, default=True
        Show progress bar

    Returns
    -------
    : GeoPandas GeoDataFrame
        columns = ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``
    """

    block_k = BlockPoissonKriging(
        semivariogram_model=semivariogram_model,
        point_support=point_support,
        kriging_type='atp',
        verbose=verbose
    )

    parsed = block_k.regularize(
        number_of_neighbors=number_of_neighbors,
        data_crs=data_crs,
        raise_when_negative_prediction=raise_when_negative_prediction,
        raise_when_negative_error=raise_when_negative_error
    )
    return parsed


class BlockPoissonKriging:
    """
    Area-to-Area, Area-to-Point, or Centroid-based Poisson Kriging
    regularization.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        The fitted variogram model.

    point_support : PointSupport
        Blocks and their point supports.

    kriging_type : str, default='ata'
        A type of Poisson Kriging operation. Available methods:
          * ``'ata'``: Area-to-Area Poisson Kriging.
          * ``'atp'``: Area-to-Point Poisson Kriging.
          * ``'cb'``: Centroid-based Poisson Kriging.

    Attributes
    ----------
    semivariogram_model : TheoreticalVariogram
        See the ``semivariogram_model`` parameter.

    point_support : PointSupport
        See the ``point_support`` parameter.

    kriging_type : str, default='ata'
        See the ``kriging_type`` parameter.

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
        Regularize ps_blocks (you should use it for data deconvolution -
        with Area-to-Point Poisson Kriging, or for data filtering -
        with Area-to-Area Poisson Kriging, or Centroid-based Poisson Kriging).
    """

    def __init__(self,
                 semivariogram_model: TheoreticalVariogram,
                 point_support: PointSupport,
                 kriging_type='ata',
                 verbose=False):

        self.kriging_type = self._check_given_kriging_type(kriging_type)

        # Check if semivariogram exists
        self.semivariogram_model = semivariogram_model
        self.point_support = point_support
        self.point_support_distance = PointSupportDistance()

        # Final GeoDataFrame with regularized data
        self.gdf = None

        # Error of regularization
        self.statistics = {}

        # Control
        self.raise_when_negative_prediction = True
        self.raise_when_negative_error = True
        self._disable_tqdm_bar = not verbose

    def regularize(self,
                   number_of_neighbors,
                   data_crs=None,
                   raise_when_negative_prediction=True,
                   raise_when_negative_error=True) -> gpd.GeoDataFrame:
        """
        Function regularizes whole dataset and creates new values and error
        maps based on the kriging type. Function does not predict unknown
        areas and areas with missing values.

        Parameters
        ----------
        number_of_neighbors : int
            The minimum number of neighbours that potentially affect a block.

        data_crs : str, default=None
            Data crs, look into: https://geopandas.org/projections.html.
            If ``None`` given then returned GeoDataFrame doesn't have a crs.

        raise_when_negative_prediction : bool, default=True
            Raise error when prediction is negative.

        raise_when_negative_error : bool, default=True
            Raise error when prediction error is negative.

        Returns
        -------
        regularized : gpd.GeoDataFrame
            Regularized set of ps_blocks:
            ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``
        """
        t_start = time.perf_counter()

        block_ids = self.point_support.unique_blocks

        interpolation_results = []
        for block_id in tqdm(block_ids, disable=self._disable_tqdm_bar):
            prediction_dict = self._interpolate(
                uid=block_id,
                n_neighbours=number_of_neighbors,
                pred_raise=raise_when_negative_prediction,
                err_raise=raise_when_negative_error
            )

            interpolation_results.extend(
                self._parse_results(
                    prediction_dict
                )
            )

        # Transform results into a dataframe
        gdf = self._to_gdf(interpolation_results, data_crs)

        # Update stats
        t_end = time.perf_counter() - t_start

        self._update_stats(gdf, t_end)

        return gdf

    def _check_given_kriging_type(self, k_type):
        """
        Checks if users provided correct kriging type.

        Parameters
        ----------
        k_type : str
            A type of Poisson Kriging operation. Available methods:
              * ``'ata'``: Area-to-Area Poisson Kriging.
              * ``'atp'``: Area-to-Point Poisson Kriging.
              * ``'cb'``: Centroid-based Poisson Kriging.

        Returns
        -------
        : str
            Kriging Type.
        """
        kriging_fns = ['ata', 'atp', 'cb']
        if k_type not in kriging_fns:
            self._raise_wrong_kriging_type_error()

        return k_type

    def _interpolate(self, uid, n_neighbours, pred_raise, err_raise) -> Dict:
        """
        Function interpolates block values using one of Poisson Kriging types.

        Parameters
        ----------
        uid : Union[Hashable, str]
            Block id - this block will be interpolated.

        n_neighbours : int
            Number of closest neighbors.

        pred_raise : bool
            Raise error when prediction is negative.

        err_raise : bool
            Raise error when prediction error is negative.

        Returns
        -------
        : dict
            For Area to Area Poisson Kriging and Centroid-based Poisson
            Kriging function returns block index, prediction, error:
            ``{"block_id": ..., "zhat": number, "sig": number}``.
            For Area to Point Poisson Kriging function returns dictionary
            with numpy array of predictions and errors:
            ``{"block": [["x", "y", "zhat", "sig"], ...]}``
        """
        # Model
        model_output = {}

        if self.kriging_type == 'ata':
            model_output = area_to_area_pk(semivariogram_model=self.semivariogram_model,
                                           point_support=self.point_support,
                                           unknown_block_index=uid,
                                           number_of_neighbors=n_neighbours,
                                           raise_when_negative_error=err_raise,
                                           raise_when_negative_prediction=pred_raise)

        elif self.kriging_type == 'atp':
            model_output = area_to_point_pk(semivariogram_model=self.semivariogram_model,
                                            point_support=self.point_support,
                                            unknown_block_index=uid,
                                            number_of_neighbors=n_neighbours,
                                            raise_when_negative_prediction=pred_raise,
                                            raise_when_negative_error=err_raise)
        elif self.kriging_type == 'cb':
            model_output = centroid_poisson_kriging(semivariogram_model=self.semivariogram_model,
                                                    point_support=self.point_support,
                                                    unknown_block_index=uid,
                                                    number_of_neighbors=n_neighbours,
                                                    raise_when_negative_prediction=pred_raise,
                                                    raise_when_negative_error=err_raise)
        else:
            self._raise_wrong_kriging_type_error()

        return model_output

    def _parse_results(self, prediction: Dict):
        """
        Function parses data to form: Regularized set of ps_blocks:
        ``['block id', 'geometry', 'reg.est', 'reg.err', 'rmse']``

        Parameters
        ----------
        prediction : dict
            For ATA and ATP: ``{"block id", "zhat", "sig"}``.
            For ATP: ``{"block id": [["x", "y", "zhat", "sig"], ...]}``

        Returns
        -------
        parsed_results : List
            Block id, point geometry, prediction, variance error,
            root mean squared error.
        """

        if self.kriging_type == 'ata' or self.kriging_type == 'cb':

            uid = prediction['block_id']
            block_real_value = self.point_support.blocks.block_real_value(
                block_id=uid
            )
            block_coordinates = self.point_support.blocks.block_coordinates(
                block_id=uid
            )
            rmse = root_mean_squared_error(prediction['zhat'],
                                           block_real_value)

            parsed_results = [
                [
                    uid, block_coordinates,
                    prediction['zhat'],
                    prediction['sig'],
                    rmse
                ]
            ]
            return parsed_results
        elif self.kriging_type == 'atp':
            # ``{"block_id": [["x", "y", "zhat", "sig"], ...]}``
            uid = list(prediction.keys())[0]
            vals = prediction[uid]

            parsed_results = []
            for row in vals:
                parsed = [
                    uid, Point(row[0], row[1]), row[2], row[3], np.nan
                ]
                parsed_results.append(parsed)
            return parsed_results
        else:
            self._raise_wrong_kriging_type_error()

    def _update_stats(self, gdf, t_delta):
        """
        Function updates ``statistics`` dictionary with processing time and RMSE.

        Parameters
        ----------
        gdf : GeoDataFrame
            Columns = ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``

        t_delta : float
            Time in seconds how long regularization has taken.
        """
        self.statistics['time (s)'] = t_delta
        self.statistics['RMSE'] = np.mean(gdf['rmse'])

        if not self._disable_tqdm_bar:
            print(f'Processing time: {t_delta} seconds')
            print(
                f'Average RMSE '
                f'(valid only for area-to-area and centroid-based '
                f'Poisson Kriging: {self.statistics["RMSE"]}'
            )

    @staticmethod
    def _raise_wrong_kriging_type_error():
        l1 = ('Provided argument is not correct. '
              'You must choose kriging type.\n')
        l2 = "'ata' - area to area Poisson Kriging,\n"
        l3 = "'atp' - area to point Poisson Kriging,\n."
        l4 = "'cb' - centroid-based Poisson Kriging."
        message = l1 + l2 + l3 + l4
        raise KeyError(message)

    @staticmethod
    def _to_gdf(list_of_results, crs=None):
        """
        Function transforms list of results into GeoDataFrame.

        Parameters
        ----------
        list_of_results : list
            Block id, point geometry, prediction, variance error,
            root mean squared error.

        crs : str, optional
            Data crs, look into: https://geopandas.org/projections.html.
            If None given then returned GeoDataFrame doesn't have a crs.

        Returns
        -------
        : GeoDataFrame
            Columns = ``['id', 'geometry', 'reg.est', 'reg.err', 'rmse']``
        """
        gdf = gpd.GeoDataFrame(list_of_results)
        gdf.columns = ['id', 'geometry', 'reg.est', 'reg.err', 'rmse']
        gdf.geometry = gdf['geometry']
        if crs is not None:
            gdf.crs = crs
        return gdf
