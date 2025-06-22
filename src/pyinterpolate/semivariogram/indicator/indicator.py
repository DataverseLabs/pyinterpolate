from copy import deepcopy
from typing import Union, List, Any
from numpy.typing import ArrayLike

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import markers
from tqdm import tqdm

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def code_indicators(ds: np.ndarray, thresholds: ArrayLike) -> np.ndarray:
    """
    Function transforms kriging values into a vector of their indicators.

    Parameters
    ----------
    ds : numpy array
        Kriging dataset [lon, lat, value]

    thresholds : List
        The list of possible thresholds.

    Returns
    -------
    ids : numpy array
        [lon, lat, bin value for thresh 0, ..., bin value for thresh n].

    """

    ids = []
    thresh_arr = np.array(thresholds)

    for row in ds:
        _r = [row[0], row[1]]
        _val = row[2]
        _indicators = _val <= thresh_arr
        _r.extend(_indicators.astype(int))
        ids.append(_r)

    ids = np.array(ids)
    return ids


def select_variogram_thresholds(ds: ArrayLike,
                                n_thresh: int) -> List[float]:
    """
    Function selects ``n_thresh`` thresholds of a sample dataset from
    its histogram, it divides histogram based on the n-quantiles.

    Parameters
    ----------
    ds : Iterable
        Data values used for interpolation.

    n_thresh : int
        The number of thresholds.

    Returns
    -------
    thresholds : List
        Thresholds used for indicator Kriging.
    """

    quantiles = np.linspace(0, 1, n_thresh+1)

    thresholds = [np.quantile(ds, q=q) for q in quantiles[1:]]

    return thresholds


class IndicatorVariogramData:
    """
    Class describes indicator variogram data.

    Parameters
    ----------
    ds : numpy array, list, tuple
        Coordinates and their values: ``(pt x, pt y, value)``

    number_of_thresholds: int
        The number of thresholds to model data.

    Attributes
    ----------
    input_array : numpy array, list, tuple
        Coordinates and their values: ``(pt x, pt y, value)``

    n_thresholds: int
        The number of thresholds to model data.

    thresholds : numpy array
        The 1D numpy array with thresholds.

    ids : numpy array
        The numpy array with
        ``[coordinate_x, coordinate_y, threshold_0, ..., threshold_n]``.

    See Also
    --------
    ExperimentalIndicatorVariogram
        Class that calculates experimental variograms for each indicator.

    """

    def __init__(self,
                 ds: Union[np.ndarray, list, tuple],
                 number_of_thresholds: int):
        if not isinstance(ds, np.ndarray):
            ds = np.array(ds)

        self.input_array = ds
        self.n_thresholds = number_of_thresholds
        self.thresholds = select_variogram_thresholds(ds[:, -1], self.n_thresholds)
        self.ids = code_indicators(ds, self.thresholds)


class ExperimentalIndicatorVariogram:
    """
    Class describes Experimental Indicator Variogram models.

    Parameters
    ----------
    ds : numpy array, list, tuple
        Coordinates and their values: ``(pt x, pt y, value)``

    number_of_thresholds: int
        The number of thresholds to model data.

    step_size : float
        The distance between lags within each points are included in
        the calculations.

    max_range : float
        The maximum range of analysis.

    custom_weights : numpy array, default=None
        Weights assigned to points, index of weight must be the same
        as index of point.

    direction : float (in range [0, 360]), default=None
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), default=1
        If ``tolerance`` is 0 then points must be placed at a single line
        with the beginning in the origin of the coordinate system and the
        direction given by y axis and direction parameter. If ``tolerance``
        is ``> 0`` then the bin is selected as an elliptical area with major
        axis pointed in the same direction as the line for 0 tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    dir_neighbors_selection_method : str, default = triangular
        Neighbors selection in a given distance. Available methods:

        * "triangle" or "t", default method where point neighbors are
          selected from triangular area,
        * "ellipse" or "e", more accurate method but also slower.

    fit : bool, default = True
        Should models be fitted in the class initialization?

    Attributes
    ----------
    ds : IndicatorVariogramData
        Prepared indicator data.

    step_size : float
        Derived from the ``step_size`` parameter.

    max_range : float
        Derived from the ``max_range`` parameter.

    custom_weights : numpy array
        Derived from the ``weights`` parameter.

    direction : float
        Derived from the ``direction`` parameter.

    tolerance : float
        Derived from the ``tolerance`` parameter.

    dir_neighbors_selection_method : str
        Derived from the ``dir_neighbors_selection_method`` parameter.

    experimental_models : List
        The ``[threshold, experimental_variogram]`` pairs.

    Methods
    -------
    fit()
        Fits indicators to experimental variograms.

    show()
        Show experimental variograms for each indicator.

    References
    ----------
    Goovaerts P. AUTO-IK: a 2D indicator kriging program for automated
    non-parametric modeling of local uncertainty
    in earth sciences. DOI: TODO
    """

    def __init__(self,
                 ds: Union[np.ndarray, list, tuple],
                 number_of_thresholds: int,
                 step_size: float,
                 max_range: float,
                 custom_weights=None,
                 custom_bins=None,
                 direction: float = None,
                 tolerance: float = None,
                 dir_neighbors_selection_method='t',
                 fit=True):

        self.ds = IndicatorVariogramData(
            ds=ds,
            number_of_thresholds=number_of_thresholds
        )

        self.step_size = step_size
        self.max_range = max_range
        self.custom_weights = custom_weights
        self.custom_bins = custom_bins
        self.direction = direction
        self.tolerance = tolerance
        self.dir_neighbors_selection_method = dir_neighbors_selection_method

        self.experimental_models = {}

        if fit:
            self.fit()

    def fit(self):
        """
        Function fits indicators to models and updates class models.
        """
        for idx, indicator in enumerate(tqdm(self.ds.thresholds)):
            _index = 2 + idx
            exp = ExperimentalVariogram(
                ds=self.ds.ids[:, [0, 1, _index]],
                step_size=self.step_size,
                max_range=self.max_range,
                custom_weights=self.custom_weights,
                custom_bins=self.custom_bins,
                direction=self.direction,
                tolerance=self.tolerance,
                dir_neighbors_selection_method=self.dir_neighbors_selection_method,
                is_semivariance=True,
                is_covariance=True
            )
            self.experimental_models[str(indicator)] = exp

    def show(self):
        """
        Function shows generated experimental variograms for each indicator.
        """
        legend = []
        plt.figure(figsize=(12, 6))

        for idx, rec in self.experimental_models.items():
            rec: ExperimentalVariogram
            plt.scatter(rec.lags, rec.semivariances)
            legend.append(f'{float(idx):.2f}')

        plt.legend(legend)
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()


class TheoreticalIndicatorVariogram:
    """
    Class models indicator variograms for all indices.

    Parameters
    ----------
    experimental_indicator_variogram : ExperimentalIndicatorVariogram
        Fitted experimanetal variograms with indicators for each threshold.

    Attributes
    ----------
    experimental_indicator_variogram : ExperimentalIndicatorVariogram
        See ``experimental_indicator_variogram`` parameter.

    theoretical_indicator_variograms : Dict
        Dictionary with fitted theoretical models for each threshold.

    Methods
    -------
    fit()
        Fits theoretical models to experimental variograms.

    show()
        Shows experimental and theoretical variograms for each threshold.

    """

    def __init__(
            self,
            experimental_indicator_variogram: ExperimentalIndicatorVariogram
    ):
        self.experimental_indicator_variogram = experimental_indicator_variogram
        self.theoretical_indicator_variograms = {}

    def fit(self,
            models_group: Union[str, list] = 'safe',
            nugget=None,
            min_nugget=0,
            max_nugget=0.5,
            number_of_nuggets=16,
            rang=None,
            min_range=0.1,
            max_range=0.5,
            number_of_ranges=16,
            sill=None,
            n_sill_values=5,
            sill_from_variance=False,
            min_sill=0.5,
            max_sill=2,
            number_of_sills=16,
            direction=None,
            error_estimator='rmse',
            deviation_weighting='equal'):
        """
        Method tries to find the optimal range, sill and model
        (function) of the theoretical semivariogram.

        Parameters
        ----------
        models_group : str or list, default='safe'
            Models group to test:

            - 'all' - the same as list with all models,
            - 'safe' - ['linear', 'power', 'spherical']
            - as a list: multiple model types to test
            - as a single model type from:
                - 'circular',
                - 'cubic',
                - 'exponential',
                - 'gaussian',
                - 'linear',
                - 'power',
                - 'spherical'.

        nugget : float, optional
            Nugget (bias) of a variogram. If given then it is
            fixed to this value.

        min_nugget : float, default = 0
            The minimum nugget as the ratio of the parameter to
            the first lag variance.

        max_nugget : float, default = 0.5
            The maximum nugget as the ratio of the parameter to
            the first lag variance.

        number_of_nuggets : int, default = 16
            How many equally spaced nuggets tested between
            ``min_nugget`` and ``max_nugget``.

        rang : float, optional
            If given, then range is fixed to this value.

        min_range : float, default = 0.1
            The minimal fraction of a variogram range,
            ``0 < min_range <= max_range``.

        max_range : float, default = 0.5
            The maximum fraction of a variogram range,
            ``min_range <= max_range <= 1``. Parameter ``max_range`` greater
            than **0.5** raises warning.

        number_of_ranges : int, default = 16
            How many bins are tested between
            ``min_range`` and ``max_range``.

        sill : float, default = None
            If given, then sill is fixed to this value.

        n_sill_values : int, default = 5
            The last n experimental semivariance records for sill estimation.
            (Used only when ``sill_from_variance`` is set to ``False``).

        sill_from_variance : bool, default = False
            Estimate sill from the variance (semivariance at distance 0).

        min_sill : float, default = 1
            The minimal fraction of the value chosen with the sill estimation
            method. The value is: for ``sill_from_values`` - the mean of
            the last ``n_sill_values`` number of experimental semivariances,
            for ``sill_from_variance`` - the experimental variogram variance.

        max_sill : float, default = 5
            The maximum fraction of the value chosen with the sill estimation
            method. The value is: for ``sill_from_values`` - the mean of
            the last ``n_sill_values`` number of experimental semivariances,
            for ``sill_from_variance`` - the experimental variogram variance.

        number_of_sills : int, default = 16
            How many equally spaced sill values are tested between
            ``min_sill`` and ``max_sill``.

        direction : float, in range [0, 360], default=None
            The direction of a semivariogram. If ``None`` given then
            semivariogram is isotropic. This parameter is required if passed
            experimental variogram is stored in a numpy array.

        error_estimator : str, default = 'rmse'
            A model error estimation method. Available options are:

            - 'rmse': Root Mean Squared Error,
            - 'mae': Mean Absolute Error,
            - 'bias': Forecast Bias,
            - 'smape': Symmetric Mean Absolute Percentage Error.

        deviation_weighting : str, default = "equal"
            The name of the method used to weight error at a given lags. Works
            only with RMSE. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger weights,
            - distant: lags that are further away have bigger weights,
            - dense: error is weighted by the number of point pairs within lag.
        """

        experimental_models = self.experimental_indicator_variogram.experimental_models

        for idx, experimental in experimental_models.items():
            theo = TheoreticalVariogram()
            theo.autofit(experimental_variogram=experimental,
                         models_group=models_group,
                         nugget=nugget,
                         min_nugget=min_nugget,
                         max_nugget=max_nugget,
                         number_of_nuggets=number_of_nuggets,
                         rang=rang,
                         min_range=min_range,
                         max_range=max_range,
                         number_of_ranges=number_of_ranges,
                         sill=sill,
                         n_sill_values=n_sill_values,
                         sill_from_variance=sill_from_variance,
                         min_sill=min_sill,
                         max_sill=max_sill,
                         number_of_sills=number_of_sills,
                         direction=direction,
                         error_estimator=error_estimator,
                         deviation_weighting=deviation_weighting,
                         return_params=False)
            self.theoretical_indicator_variograms[idx] = theo

    def show(self, subplots: bool = False):
        """
        Method plots experimental and theoretical variograms.

        Parameters
        ----------
        subplots : bool, default = False
            If ``True`` then each indicator variogram is plotted on a separate
            plot. Otherwise, all variograms are plotted on a single plot.
        """

        if subplots:
            number_of_subplots = len(
                list(self.theoretical_indicator_variograms.keys())
            )
            fig, axes = plt.subplots(number_of_subplots,
                                     sharey=True,
                                     sharex=True,
                                     constrained_layout=True)
            axes: Any  # type checker disabled
            idx_val = 0
            for _key, _item in self.theoretical_indicator_variograms.items():
                _item: TheoreticalVariogram
                axes[idx_val].scatter(_item.lags,
                                      _item.experimental_semivariances,
                                      marker='x', c='#101010', alpha=0.2)

                axes[idx_val].plot(_item.lags,
                                   _item.yhat,
                                   '--',
                                   color='#fc8d62')
                axes[idx_val].set_title('Threshold: ' + f'{float(_key):.2f}')

                idx_val = idx_val + 1

        else:
            # Plot everything on a single plot
            plt.figure(figsize=(10, 10))
            legend = []
            markers_list = list(
                markers.MarkerStyle.filled_markers
            )  # Only filled
            colors_list = [
                '#66c2a5',
                '#fc8d62',
                '#8da0cb'
            ]
            cidx = 0
            mlist = deepcopy(markers_list)
            _marker_idx = 1
            for _key, _item in self.theoretical_indicator_variograms.items():
                _item: TheoreticalVariogram
                try:
                    plt.scatter(_item.lags,
                                _item.experimental_semivariances,
                                marker=mlist.pop(0),
                                alpha=0.4,
                                c=colors_list[cidx],
                                edgecolors='black')
                except IndexError:
                    mlist = deepcopy(markers_list)
                    cidx = cidx + 1
                    plt.scatter(_item.lags,
                                _item.experimental_semivariances,
                                marker=mlist.pop(0),
                                alpha=0.4,
                                c=colors_list[cidx],
                                edgecolors='black')

                no_items = len(_item.lags)
                plt.annotate(
                    str(_marker_idx),
                    [
                        _item.lags[no_items - 1],
                        _item.experimental_semivariances[no_items - 1]
                    ]
                )

                legend.append(
                    str(_marker_idx) + ' | T: ' + f'{float(_key):.2f}'
                )
                _marker_idx = _marker_idx + 1
            plt.legend(legend)

        plt.show()
