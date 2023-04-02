"""
Perform indicator kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

Bibliography
------------
[1] P. Goovaerts, AUTO-IK: A 2D indicator kriging program for the automated non-parametric modeling of local
    uncertainty in earth sciences, Computers & Geosciences, Volume 35, Issue 6, 2009, Pages 1255-1270, ISSN 0098-3004,
    https://doi.org/10.1016/j.cageo.2008.08.014.

Indicator kriging is performed for each threshold using four types of destination geography:
# TODO: helper methods to perform operations below:
1) grid of points specified by the user,
2) rectangular grid,
3) sampled locations (cross-validation option), and
4) set of test locations (jack-knife option).
"""
from typing import Tuple, Dict

import numpy as np
from tqdm import tqdm

from scipy.interpolate import UnivariateSpline

from pyinterpolate.kriging.models.point.ordinary_kriging import ordinary_kriging
from pyinterpolate.kriging.models.point.simple_kriging import simple_kriging
from pyinterpolate.variogram.indicator.indicator_variogram import IndicatorVariograms


class IndicatorKriging:
    """
    Class performs indicator kriging.

    Parameters
    ----------
    datapoints : numpy ndarray
        The known locations ``[x, y, value]``.

    indicator_variograms : IndicatorVariograms
        Modeled variograms for each threshold.

    unknown_locations : numpy ndarray
        Points where we want to estimate value ``(x, y) <-or-> (lon, lat)``.

    kriging_type : str, default = 'ok'
        Type of kriging to perform. Possible values: 'ok' - ordinary kriging, 'sk' - simple kriging.

    process_mean : float
        The mean value of a process over a study area. Should be know before processing. That's why Simple
        Kriging has a limited number of applications. You must have multiple samples and well-known area to
        know this parameter.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to
        ``True`` if you don't know what are you doing. This parameter can be useful when you have clusters in
        your dataset, that can lead to singular or near-singular matrix creation.

    get_expected_values : bool, default=True
        If ``True`` then expected values and variances are calculated.

    Attributes
    ----------
    thresholds : numpy ndarray
        Thresholds used for indicator kriging.

    coordinates : numpy ndarray
        Coordinates of unknown locations.

    indicator_predictions : numpy ndarray
        Indicator kriging predictions for each threshold and each unknown location.

    expected_values : numpy ndarray
        Expected values derived from ``indicator_predictions`` for each unknown location.

    variances : numpy ndarray
        Variances derived from ``indicator_predictions`` for each unknown location.

    Methods
    -------
    get_indicator_maps()
        Returns dictionary with thresholds and indicator maps for each of them.

    get_expected_values()
        Returns two arrays: one array with coordinates and expected values, and the second
        with coordinates and variances.
    """

    def __init__(self,
                 datapoints: np.ndarray,
                 indicator_variograms: IndicatorVariograms,
                 unknown_locations: np.ndarray,
                 kriging_type: str = 'ok',
                 process_mean: float = None,
                 neighbors_range=None,
                 no_neighbors=4,
                 use_all_neighbors_in_range=False,
                 allow_approximate_solutions=False,
                 get_expected_values=True):

        self.thresholds = np.array(list(indicator_variograms.theoretical_indicator_variograms.keys())).astype(float)
        self.coordinates = unknown_locations.copy()

        self.indicator_predictions = self._estimate(datapoints,
                                                    indicator_variograms,
                                                    unknown_locations,
                                                    kriging_type,
                                                    process_mean,
                                                    neighbors_range,
                                                    no_neighbors,
                                                    use_all_neighbors_in_range,
                                                    allow_approximate_solutions)

        self.expected_values = None
        self.variances = None
        if get_expected_values:
            self.expected_values, self.variances = self._get_expected_values()

    def get_indicator_maps(self) -> Dict:
        """
        Method returns indicator map for each threshold.

        Returns
        -------
        indicator_maps : Dict
            Indicator map for each threshold.
        """
        indicator_maps = {}
        for idx, threshold in enumerate(self.thresholds):
            indicator_maps[threshold] = np.column_stack(
                [self.coordinates[:, 0],
                self.coordinates[:, 1],
                self.indicator_predictions[:, idx]]
            )

        return indicator_maps

    def get_expected_values_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method returns expected values and variances for each threshold.

        Returns
        -------
        expected_values, variances : numpy ndarray, numpy ndarray
            Expected values and variances.
        """
        if self.expected_values is None:
            self._get_expected_values()

        # Expected values map
        expected_values_map = np.column_stack([self.coordinates[:, 0], self.coordinates[:, 1], self.expected_values])

        # Variances map
        variances_map = np.column_stack([self.coordinates[:, 0], self.coordinates[:, 1], self.variances])
        return expected_values_map, variances_map

    def _estimate(self,
                  datapoints: np.ndarray,
                  indicator_variograms: IndicatorVariograms,
                  unknown_locations: np.ndarray,
                  kriging_type: str = 'ok',
                  process_mean: float = None,
                  neighbors_range=None,
                  no_neighbors=4,
                  use_all_neighbors_in_range=False,
                  allow_approximate_solutions=False) -> np.ndarray:
        """
        Method estimates probabilities of a location becoming within a given threshold.

        Parameters
        ----------
        datapoints : numpy ndarray
            The known locations ``[x, y, value]``.

        indicator_variograms : IndicatorVariograms
            Modeled variograms for each threshold.

        unknown_locations : numpy ndarray
            Points where we want to estimate value ``(x, y) <-or-> (lon, lat)``.

        kriging_type : str, default = 'ok'
            Type of kriging to perform. Possible values: 'ok' - ordinary kriging, 'sk' - simple kriging.

        process_mean : float
            The mean value of a process over a study area. Should be know before processing. That's why Simple
            Kriging has a limited number of applications. You must have multiple samples and well-known area to
            know this parameter.

        neighbors_range : float, default=None
            The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
            the ``theoretical_model`` ``rang`` attribute.

        no_neighbors : int, default = 4
            The number of the **n-closest neighbors** used for interpolation.

        use_all_neighbors_in_range : bool, default = False
            ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
            ``number_of_neighbors`` parameter then take all of them anyway.

        allow_approximate_solutions : bool, default=False
            Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to
            ``True`` if you don't know what are you doing. This parameter can be useful when you have clusters in
            your dataset, that can lead to singular or near-singular matrix creation.

        Returns
        -------
        indicator_predictions : numpy array
            The indicator probabilities where each column is a threshold.

        Raises
        ------
        ValueError
            Wrong Kriging type, allowed types: 'ok' or 'sk'
        """
        predictions_all = []

        for _key, _item in tqdm(indicator_variograms.theoretical_indicator_variograms.items()):

            indicator_points = datapoints.copy()
            indicator_points[:, -1] = (indicator_points[:, -1] <= float(_key)).astype(int)

            predictions = []

            for _point in unknown_locations:
                if kriging_type == 'ok':
                    _pred_arr = ordinary_kriging(
                        theoretical_model=_item,
                        known_locations=indicator_points,
                        unknown_location=_point,
                        neighbors_range=neighbors_range,
                        no_neighbors=no_neighbors,
                        use_all_neighbors_in_range=use_all_neighbors_in_range,
                        allow_approximate_solutions=allow_approximate_solutions
                    )

                elif kriging_type == 'sk':
                    _pred_arr = simple_kriging(
                        theoretical_model=_item,
                        known_locations=indicator_points,
                        unknown_location=_point,
                        process_mean=process_mean,
                        neighbors_range=neighbors_range,
                        no_neighbors=no_neighbors,
                        use_all_neighbors_in_range=use_all_neighbors_in_range,
                        allow_approximate_solutions=allow_approximate_solutions
                    )

                else:
                    raise ValueError('Kriging type not supported. Please choose from: '
                                     '"ok" - ordinary kriging, "sk" - simple kriging.')

                predictions.append(_pred_arr[0])

            predictions_all.append(predictions)

        indicator_predictions = np.column_stack(predictions_all)

        indicator_predictions = self._clean_probabilities(indicator_predictions)

        return indicator_predictions

    def _get_expected_values(self, ccdf_density=100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function gets expected values and their variance for each point based on the ccdf function.

        Parameters
        ----------
        ccdf_density : int, default = 100
            The number of points used to interpolate expected value from the ccdf function.

        Returns
        -------
        expected_values, expected_variances : Tuple
            The expected value and the expected variance for each point.
        """

        expected_values = []
        expected_variances = []

        for row in self.indicator_predictions:
            old_indices = self.thresholds.copy()
            new_length = ccdf_density
            new_indices = np.linspace(old_indices.min(), old_indices.max(), new_length)
            spl = UnivariateSpline(old_indices, row, s=0)
            new_tvals = spl(new_indices)

            cdf = new_tvals.cumsum() / new_tvals.sum()
            ccdf = 1 - cdf
            expected_c = np.mean(ccdf)
            expected_c_index = np.searchsorted(np.flip(ccdf), expected_c)
            expected_c_index = len(ccdf) - expected_c_index
            expected_value = new_indices[expected_c_index]

            expected_values.append(expected_value)

            # Variance
            expected_variance = np.mean(
                np.abs(new_tvals - expected_value) ** 2
            )
            expected_variances.append(expected_variance)

        return np.array(expected_values), np.array(expected_variances)

    def _clean_probabilities(self, array: np.ndarray):
        """
        Function normalizes and transform given probabilities.

        Parameters
        ----------
        array : numpy array
            Array which rows are transformed.

        Returns
        -------
        transformed : numpy array
            Transformed (normalized along rows) array.
        """
        arr = self.__cut_to_zero_one(array).copy()
        return arr

    @staticmethod
    def __cut_to_zero_one(arr):
        arr[arr > 1] = 1
        arr[arr < 0] = 0
        return arr
