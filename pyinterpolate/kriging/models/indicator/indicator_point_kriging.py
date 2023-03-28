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
1) grid of points specified by the user,
2) rectangular grid,
3) sampled locations (cross-validation option), and
4) set of test locations (jack-knife option).
"""
from typing import Dict

import numpy as np

from pyinterpolate.kriging.models.point.ordinary_kriging import ordinary_kriging
from pyinterpolate.kriging.models.point.simple_kriging import simple_kriging
from pyinterpolate.variogram.indicator.indicator_variogram import IndicatorVariograms


class IndicatorKriging:

    def __init__(self,
                 datapoints: np.ndarray,
                 indicator_variograms: IndicatorVariograms,
                 unknown_locations: np.ndarray,
                 kriging_type: str = 'ok',
                 process_mean: float = None,
                 neighbors_range=None,
                 no_neighbors=4,
                 use_all_neighbors_in_range=False):
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
        """

        self.indicator_outputs = self._estimate(datapoints,
                                                indicator_variograms,
                                                unknown_locations,
                                                kriging_type,
                                                process_mean,
                                                neighbors_range,
                                                no_neighbors,
                                                use_all_neighbors_in_range)
        
        self.raw_probabilities = self._get_raw_probabilities()


    def _estimate(self, known_points, variograms, unknowns, ktype, pmean, nrange, no_neigh, all_in_rng) -> Dict:
        """
        Function performs indicator kriging for each threshold.
        """
        indicator_outputs = {}

        for _key, _item in variograms.theoretical_indicator_variograms.items():
            predictions = []
            errors = []

            for _point in unknowns:
                if ktype == 'ok':
                    _pred_arr = ordinary_kriging(
                        theoretical_model=_item,
                        known_locations=known_points,
                        unknown_location=_point,
                        neighbors_range=nrange,
                        no_neighbors=no_neigh,
                        use_all_neighbors_in_range=all_in_rng
                    )

                elif ktype == 'sk':
                    _pred_arr = simple_kriging(
                        theoretical_model=_item,
                        known_locations=known_points,
                        unknown_location=_point,
                        process_mean=pmean,
                        neighbors_range=nrange,
                        no_neighbors=no_neigh,
                        use_all_neighbors_in_range=all_in_rng
                    )

                else:
                    raise ValueError('Kriging type not supported. Please choose from: '
                                     '"ok" - ordinary kriging, "sk" - simple kriging.')

                predictions.append(_pred_arr[0])
                errors.append(_pred_arr[1])

            indicator_outputs[_key] = {
                'predictions': np.array(predictions),
                'errors': np.array(errors)
            }

        return indicator_outputs