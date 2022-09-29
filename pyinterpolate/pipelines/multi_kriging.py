"""
Block data interpolation with a different kriging techniques.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

TODO
----
* tests
"""
from typing import Union, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

from tqdm import tqdm

from pyinterpolate.kriging.models.block import area_to_area_pk, centroid_poisson_kriging
from pyinterpolate.kriging.point_kriging import kriging
from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport
from pyinterpolate.processing.transform.transform import transform_ps_to_dict, transform_blocks_to_numpy
from pyinterpolate.variogram import TheoreticalVariogram


class BlockToBlockKrigingComparison:
    """
    Class compares different block kriging models and techniques.

    Parameters
    ----------
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

    no_of_neighbors : int, default = 16
        The maximum number of n-closest neighbors used for interpolation.

    neighbors_range : float, default = None
        Maximum distance where we search for point neighbors. If None given then range is selected from
        the ``theoretical_model`` ``rang`` attribute. If algorithms picks less neighbors than ``no_of_neighbors``
        within the range then additional points are selected outside the ``neighbors_range``.

    simple_kriging_mean : float, default = None
        The mean value of a process over a study area. Should be known before processing. If not provided then
        Simple Kriging estimator is skipped.

    raise_when_negative_prediction : bool, default = True
        Raise error when prediction is negative.

    raise_when_negative_error : bool, default=True
        Raise error when prediction error is negative.

    training_set_frac : float, default = 0.8
        How many values sampled as a known points set in each iteration. Could be any fraction within ``(0:1)`` range.

    allow_approx_solutions : bool, default = False
        Allows the approximation of kriging weights based on the OLS algorithm. Not recommended to set to ``True`` if
        you don't know what you are doing!

    iters : int, default = 20
        How many tests to perform over random samples of a data.

    Attributes
    ----------
    variogram : TheoreticalVariogram
        See ``variogram`` parameter.

    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
        See ``blocks`` parameter.

    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
        See ``point_support`` parameter.

    no_of_neighbors : int
        See ``no_of_neighbors`` parameter.

    neighbors_range : float
        See ``neighbors_range`` parameter.

    simple_kriging_mean : float
        See ``simple_kriging_mean`` parameter.

    raise_when_negative_prediction : bool, default = True
        See ``raise_when_negative_prediction`` parameter.

    raise_when_negative_error : bool, default=True
        See ``raise_when_negative_error`` parameter.

    training_set_frac : float
        See ``training_set_frac`` parameter.

    iters : int
        See ``iters`` parameter.

    common_indexes : Set
        Indexes that are common for blocks and point support.

    training_set_indexes : List[List]
        List of lists with indexes used in a random sampling for a training set.

    results : Dict
        Results for each type of Block Kriging method.

    Methods
    -------
    run_tests()
        Compares different types of Kriging, returns Dict with the mean root mean squared error of each iteration.
    """

    def __init__(self,
                 variogram: TheoreticalVariogram,
                 blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                 point_support: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                 no_of_neighbors: int = 16,
                 neighbors_range: float = None,
                 simple_kriging_mean: float = None,
                 raise_when_negative_prediction=True,
                 raise_when_negative_error=True,
                 training_set_frac=0.8,
                 allow_approx_solutions=False,
                 iters=20):

        self.variogram = variogram
        self.blocks = blocks
        self.point_support = point_support
        self.no_of_neighbors = no_of_neighbors
        self.neighbors_range = neighbors_range
        self.simple_kriging_mean = simple_kriging_mean
        self.raise_when_negative_prediction = raise_when_negative_prediction
        self.raise_when_negative_error = raise_when_negative_error
        self.training_set_frac = training_set_frac
        self.allow_approx_solutions = allow_approx_solutions
        self.iters = iters
        self.common_indexes = None
        self.training_set_indexes = []

        self.results = {
            'PK-ata': np.nan,
            'PK-centroid': np.nan,
            'OK': np.nan,
            'SK': np.nan
        }

    def _divide_train_test(self, arr_bl, dict_ps) -> List:
        """
        Function divides data into a training and a test set.

        Parameters
        ----------
        arr_bl : numpy array

        dict_ps : Dict

        Returns
        -------
        sets : List
               [training_block_arr, training_point_support, test_block_arr, test_point_support]
        """

        # Get IDs - only those that are present in the blocks and point support!
        common_ids = set(dict_ps.keys()) & set(arr_bl[:, 0])
        self.common_indexes = common_ids.copy()

        # Get IDs for a training and test sets
        number_of_samples = int(len(self.common_indexes) * self.training_set_frac)
        training_set_ids = np.random.choice(list(self.common_indexes), number_of_samples, replace=False)
        test_set_ids = [x for x in self.common_indexes if x not in training_set_ids]

        training_block_arr = np.array(
            [x for x in arr_bl if x[0] in training_set_ids]
        )
        test_block_arr = np.array(
            [x for x in arr_bl if x[0] in test_set_ids]
        )

        training_point_support = {}
        test_point_support = {}

        for idx, value in dict_ps.items():
            if idx in training_set_ids:
                training_point_support[idx] = value
            else:
                test_point_support[idx] = value

        result = [training_block_arr, training_point_support, test_block_arr, test_point_support]

        return result

    def _run_pk_ata(self, training_areas, training_points, test_areas, test_points):
        # Poisson Kriging model Area-to-area
        pk_preds = []
        for unknown_area in test_areas:
            uidx = unknown_area[0]
            uval = unknown_area[-1]
            result = area_to_area_pk(semivariogram_model=self.variogram,
                                     blocks=training_areas,
                                     point_support=training_points,
                                     unknown_block=unknown_area[:-1],
                                     unknown_block_point_support=test_points[uidx],
                                     number_of_neighbors=self.no_of_neighbors,
                                     raise_when_negative_prediction=self.raise_when_negative_prediction,
                                     raise_when_negative_error=self.raise_when_negative_error)

            err = np.sqrt(
                (uval - result[1]) ** 2
            )
            pk_preds.append(err)

        return np.mean(pk_preds)

    def _run_pk_centroid(self, training_areas, training_points, test_areas, test_points):
        # Poisson Kriging centroid based approach
        pk_preds = []
        for unknown_area in test_areas:
            uidx = unknown_area[0]
            uval = unknown_area[-1]

            result = centroid_poisson_kriging(semivariogram_model=self.variogram,
                                              blocks=training_areas,
                                              point_support=training_points,
                                              unknown_block=unknown_area[:-1],
                                              unknown_block_point_support=test_points[uidx],
                                              number_of_neighbors=self.no_of_neighbors,
                                              raise_when_negative_prediction=self.raise_when_negative_prediction,
                                              raise_when_negative_error=self.raise_when_negative_error)

            err = np.sqrt(
                (uval - result[1]) ** 2
            )
            pk_preds.append(err)

        return np.mean(pk_preds)

    def _run_ok_point(self, training_areas, test_areas):
        # Ordinary Kriging - only blocks data
        test_pts = test_areas[:, 1:-1]
        k_obs = training_areas[:, 1:]

        results = kriging(observations=k_obs,
                          theoretical_model=self.variogram,
                          points=test_pts,
                          how='ok',
                          neighbors_range=self.neighbors_range,
                          no_neighbors=self.no_of_neighbors,
                          allow_approx_solutions=self.allow_approx_solutions)

        mean_error = np.mean(np.sqrt(
            (test_areas[:, -1] - results[:, 0]) ** 2
        ))

        return mean_error

    def _run_sk_point(self, training_areas, test_areas):
        # Simple Kriging

        test_pts = test_areas[:, 1:-1]
        k_obs = training_areas[:, 1:]

        results = kriging(observations=k_obs,
                          theoretical_model=self.variogram,
                          points=test_pts,
                          how='sk',
                          neighbors_range=self.neighbors_range,
                          no_neighbors=self.no_of_neighbors,
                          sk_mean=self.simple_kriging_mean,
                          allow_approx_solutions=self.allow_approx_solutions)

        mean_error = np.mean(np.sqrt(
            (test_areas[:, -1] - results[:, 0]) ** 2
        ))

        return mean_error

    def run_tests(self):
        """
        Method compares ordinary, simple, area-to-area and centroid-based block Poisson Kriging.
        """

        # Prepare data for testing

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

        pk_ata_results = []
        pk_cb_results = []
        ok_results = []
        sk_results = []

        for _ in tqdm(range(self.iters)):
            # Generate training and test set
            sets = self._divide_train_test(arr_bl, dict_ps)

            # Poisson Kriging
            pk_res = self._run_pk_ata(*sets)
            pk_ata_results.append(pk_res)

            # Centroid-based PK
            cb_res = self._run_pk_centroid(*sets)
            pk_cb_results.append(cb_res)

            # Ordinary Kriging
            ok_res = self._run_ok_point(sets[0], sets[2])
            ok_results.append(ok_res)

            # Simple Kriging
            if self.simple_kriging_mean is None:
                pass
            else:
                sk_res = self._run_sk_point(sets[0], sets[2])
                sk_results.append(sk_res)

        # Mean of values

        pk_rmse = np.mean(pk_ata_results)
        ck_rmse = np.mean(pk_cb_results)
        ok_rmse = np.mean(ok_results)

        # Simple Kriging case
        if self.simple_kriging_mean is None:
            sk_rmse = np.nan
        else:
            sk_rmse = np.mean(sk_results)

        # Update dict
        self.results['PK-ata'] = float(pk_rmse)
        self.results['PK-centroid'] = float(ck_rmse)
        self.results['OK'] = float(ok_rmse)
        self.results['SK'] = float(sk_rmse)

        return self.results
