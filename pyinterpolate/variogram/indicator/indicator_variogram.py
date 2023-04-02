from copy import deepcopy
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import markers
from tqdm import tqdm

from pyinterpolate.processing.transform.statistics import select_variogram_thresholds
from pyinterpolate.processing.transform.transform import code_indicators
from pyinterpolate.variogram.empirical.experimental_variogram import ExperimentalVariogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


class IndicatorVariogramData:
    """
    Class describes indicator variogram data.

    Parameters
    ----------
    input_array : numpy array, list, tuple
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
        The numpy array with ``[coordinate_x, coordinate_y, threshold_0, ..., threshold_n]``.

    See Also
    --------
    ExperimentalIndicatorVariogram
        Class that calculates experimental variograms for each indicator.

    """

    def __init__(self,
                 input_array: Union[np.ndarray, list, tuple],
                 number_of_thresholds: int):
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        self.input_array = input_array
        self.n_thresholds = number_of_thresholds
        self.thresholds = select_variogram_thresholds(input_array[:, -1], self.n_thresholds)
        self.ids = code_indicators(input_array, self.thresholds)


class ExperimentalIndicatorVariogram:
    """
    Class describes Experimental Indicator Variogram models.

    Parameters
    ----------
    input_array : numpy array, list, tuple
        Coordinates and their values: ``(pt x, pt y, value)``

    number_of_thresholds: int
        The number of thresholds to model data.

    step_size : float
        The distance between lags within each points are included in the calculations.

    max_range : float
        The maximum range of analysis.

    weights : numpy array, default=None
        Weights assigned to points, index of weight must be the same as index of point.

    direction : float (in range [0, 360]), default=None
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), default=1
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y axis and direction parameter. If ``tolerance`` is ``> 0``
        then the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    method : str, default = triangular
        The method used for neighbors selection. Available methods:

        * "triangle" or "t", default method where a point neighbors are selected from a triangular area,
        * "ellipse" or "e", the most accurate method but also the slowest one.

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

    weights : numpy array
        Derived from the ``weights`` parameter.

    direction : float
        Derived from the ``direction`` parameter.

    tolerance : float
        Derived from the ``tolerance`` parameter.

    method : str
        Derived from the ``method`` parameter.

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
    Goovaerts P. AUTO-IK: a 2D indicator kriging program for automated non-parametric modeling of local uncertainty
    in earth sciences. DOI: TODO
    """

    def __init__(self,
                 input_array: Union[np.ndarray, list, tuple],
                 number_of_thresholds: int,
                 step_size: float,
                 max_range: float,
                 weights=None,
                 direction: float = None,
                 tolerance: float = 1.0,
                 method='t',
                 fit=True):

        self.ds = IndicatorVariogramData(input_array=input_array, number_of_thresholds=number_of_thresholds)

        self.step_size = step_size
        self.max_range = max_range
        self.weights = weights
        self.direction = direction
        self.tolerance = tolerance
        self.method = method

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
                input_array=self.ds.ids[:, [0, 1, _index]],
                step_size=self.step_size,
                max_range=self.max_range,
                weights=self.weights,
                direction=self.direction,
                tolerance=self.tolerance,
                method=self.method,
                is_semivariance=True,
                is_covariance=True,
                is_variance=True
            )
            self.experimental_models[str(indicator)] = exp

    def show(self):
        """
        Function shows generated experimental variograms for each indicator.
        """
        legend = []
        plt.figure(figsize=(12, 6))

        for idx, rec in self.experimental_models.items():
            plt.scatter(rec.lags, rec.experimental_semivariances)
            legend.append(f'{float(idx):.2f}')

        plt.legend(legend)
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()


class IndicatorVariograms:
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

    def __init__(self, experimental_indicator_variogram: ExperimentalIndicatorVariogram):
        self.experimental_indicator_variogram = experimental_indicator_variogram
        self.theoretical_indicator_variograms = {}

    def fit(self,
            model_type: str = 'linear',
            nugget=0,
            rang=None,
            min_range=0.1,
            max_range=0.5,
            number_of_ranges=64,
            sill=None,
            min_sill=0.5,
            max_sill=1.5,
            number_of_sills=64,
            direction=None,
            error_estimator='rmse',
            deviation_weighting='equal',
            auto_update_attributes=True,
            warn_about_set_params=True,
            verbose=False):
        """
        Method tries to find the optimal range, sill and model (function) of the theoretical semivariogram.

        Parameters
        ----------
        model_type : str, default = "linear"
            The name of a modeling function. Available models:

            - 'basic' : linear and spherical models are tested,
            - 'exponential',
            - 'gaussian',
            - 'linear',
            - 'power',
            - 'spherical'.

        nugget : float, default = 0
            Nugget (bias) of a variogram. Default value is 0.

        rang : float, optional
            If given, then range is fixed to this value.

        min_range : float, default = 0.1
            The minimal fraction of a variogram range, ``0 < min_range <= max_range``.

        max_range : float, default = 0.5
            The maximum fraction of a variogram range, ``min_range <= max_range <= 1``. Parameter ``max_range`` greater
            than **0.5** raises warning.

        number_of_ranges : int, default = 64
            How many equally spaced ranges are tested between ``min_range`` and ``max_range``.

        sill : float, default = None
            If given, then sill is fixed to this value.

        min_sill : float, default = 0
            The minimal fraction of the variogram variance at lag 0 to find a sill, ``0 <= min_sill <= max_sill``.

        max_sill : float, default = 1
            The maximum fraction of the variogram variance at lag 0 to find a sill. It *should be* lower or equal to 1.
            It is possible to set it above 1, but then warning is printed.

        number_of_sills : int, default = 64
            How many equally spaced sill values are tested between ``min_sill`` and ``max_sill``.

        direction : float, in range [0, 360], default=None
            The direction of a semivariogram. If ``None`` given then semivariogram is isotropic. This parameter is
            required if passed experimental variogram is stored in a numpy array.

        error_estimator : str, default = 'rmse'
            A model error estimation method. Available options are:

            - 'rmse': Root Mean Squared Error,
            - 'mae': Mean Absolute Error,
            - 'bias': Forecast Bias,
            - 'smape': Symmetric Mean Absolute Percentage Error.

        deviation_weighting : str, default = "equal"
            The name of a method used to weight error at a given lags. Works only with RMSE. Available methods:

            - equal: no weighting,
            - closest: lags at a close range have bigger weights,
            - distant: lags that are further away have bigger weights,
            - dense: error is weighted by the number of point pairs within a lag.

        auto_update_attributes : bool, default = True
            Update sill, range, model type and nugget based on the best model.

        warn_about_set_params: bool, default=True
            Should class invoke warning if model parameters has been set during initialization?

        verbose : bool, default = False
            Show iteration results.


        Returns
        -------
        model_attributes : Dict
            Attributes dict:

            >>> {
            ...     'model_type': model_name,
            ...     'sill': model_sill,
            ...     'range': model_range,
            ...     'nugget': model_nugget,
            ...     'error_type': type_of_error_metrics,
            ...     'error value': error_value
            ... }

        Warns
        -----
        SillOutsideSafeRangeWarning
            Warning printed when ``max_sill > 1``.

        RangeOutsideSafeDistanceWarning
            Warning printed when ``max_range > 0.5``.

        Warning
            Model parameters were given during initilization but program is forced to fit the new set of parameters.

        Warning
            Passed ``experimental_variogram`` is a numpy array and ``direction`` parameter is ``None``.

        Raises
        ------
        ValueError
            Raised when ``sill < 0`` or ``range < 0`` or ``range > 1``.

        KeyError
            Raised when wrong model name(s) are provided by the users.

        KeyError
            Raised when wrong error type is provided by the users.
        """

        experimental_models = self.experimental_indicator_variogram.experimental_models

        if model_type == 'basic':
            model_type = ['linear', 'spherical']

        for idx, experimental in experimental_models.items():
            theo = TheoreticalVariogram()
            theo.autofit(experimental_variogram=experimental,
                         model_types=model_type,
                         nugget=nugget,
                         rang=rang,
                         min_range=min_range,
                         max_range=max_range,
                         number_of_ranges=number_of_ranges,
                         sill=sill,
                         min_sill=min_sill,
                         max_sill=max_sill,
                         number_of_sills=number_of_sills,
                         direction=direction,
                         error_estimator=error_estimator,
                         deviation_weighting=deviation_weighting,
                         auto_update_attributes=auto_update_attributes,
                         warn_about_set_params=warn_about_set_params,
                         verbose=verbose,
                         return_params=False)
            self.theoretical_indicator_variograms[idx] = theo

    def show(self, subplots: bool = False):
        """
        Method plots experimental and theoretical variograms.

        Parameters
        ----------
        subplots : bool, default = False
            If ``True`` then each indicator variogram is plotted on a separate plot. Otherwise, all variograms are
            plotted on a scatter single plot.
        """

        if subplots:
            number_of_subplots = len(list(self.theoretical_indicator_variograms.keys()))
            fig, axes = plt.subplots(number_of_subplots, sharey=True, sharex=True, constrained_layout=True)
            idx_val = 0
            for _key, _item in self.theoretical_indicator_variograms.items():
                axes[idx_val].scatter(_item.experimental_array[:, 0],
                                      _item.experimental_array[:, 1],
                                      marker='x', c='#101010', alpha=0.2)

                axes[idx_val].plot(_item.lags, _item.fitted_model[:, 1], '--', color='#fc8d62')
                axes[idx_val].set_title('Threshold: ' + f'{float(_key):.2f}')

                idx_val = idx_val + 1

        else:
            # Plot everything on a single plot
            plt.figure(figsize=(10, 10))
            legend = []
            markers_list = list(markers.MarkerStyle.filled_markers)  # Only filled
            colors_list = [
                '#66c2a5',
                '#fc8d62',
                '#8da0cb'
            ]
            cidx = 0
            mlist = deepcopy(markers_list)
            _marker_idx = 1
            for _key, _item in self.theoretical_indicator_variograms.items():
                try:
                    plt.scatter(_item.experimental_array[:, 0],
                                _item.experimental_array[:, 1],
                                marker=mlist.pop(0),
                                alpha=0.4, c=colors_list[cidx], edgecolors='black')
                except IndexError:
                    mlist = deepcopy(markers_list)
                    cidx = cidx + 1
                    plt.scatter(_item.experimental_array[:, 0],
                                _item.experimental_array[:, 1],
                                marker=mlist.pop(0),
                                alpha=0.4, c=colors_list[cidx], edgecolors='black')

                no_items = len(_item.experimental_array[:, 0])
                plt.annotate(
                    str(_marker_idx),
                    [
                        _item.experimental_array[:, 0][no_items - 1],
                        _item.experimental_array[:, 1][no_items - 1]
                    ]
                )

                legend.append(str(_marker_idx) + ' | T: ' + f'{float(_key):.2f}')
                _marker_idx = _marker_idx + 1
            plt.legend(legend)

        plt.show()
