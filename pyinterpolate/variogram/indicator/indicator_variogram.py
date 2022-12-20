from typing import Union

import numpy as np
from matplotlib import pyplot as plt
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

    See Also
    --------

    Examples
    --------

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

        self.experimental_models = []

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
                is_covariance=False,
                is_variance=False
            )

            self.experimental_models.append(
                [indicator, exp]
            )

    def show(self):
        """
        Function shows generated experimental variograms for each indicator.
        """
        legend = []
        plt.figure(figsize=(12, 6))

        lags = self.experimental_models[0][1].lags

        for rec in self.experimental_models:
            exp = rec[1]
            lag_name = rec[0]
            plt.scatter(lags, exp.experimental_semivariances)
            legend.append(f'{lag_name:.2f}')

        plt.legend(legend)
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()


class IndicatorVariograms:
    """
    Class models indicator variograms for all indices.
    """

    def __init__(self):
        pass

    def model(self):
        pass

    def show(self):
        pass