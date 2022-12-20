"""
Class for Experimental Variogram and helper class for DirectionalVariograms.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Union, Dict, Type

import matplotlib.pyplot as plt
import numpy as np
from numpy import nan
from prettytable import PrettyTable

from pyinterpolate.variogram.empirical.covariance import calculate_covariance
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance
from pyinterpolate.variogram.utils.exceptions import validate_plot_attributes_for_experimental_variogram_class


class DirectionalVariogram:
    """
    Class prepares four directional variograms and one isotropic variogram.

    Parameters
    ----------
    input_array : numpy array, list, tuple
        * As a ``list`` and ``numpy array``: coordinates and their values: ``(pt x, pt y, value)``,
        * as a ``dict``: ``polyset = {'points': numpy array with coordinates and their values}``,
        * as a ``Blocks``: ``Blocks.polyset['points']``.

    step_size : float
        The distance between lags within each the points are included in the calculations.

    max_range : float
        The maximum range of analysis.

    weights : numpy array, default=None
        Weights assigned to the points, index of weight must be the same as index of point.

    tolerance : float (in range [0, 1]), default=0.2
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

    Attributes
    ----------
    ds : numpy array
        See the ``input_array`` parameter.

    step_size : float
        See the ``step_size`` parameter.

    max_range : float
        See the ``max_range`` parameter.

    tolerance : float
        See the ``tolerance`` parameter.

    weights : float
        See the ``weights`` parameter.

    directions : Dict
        Dictionary where keys are directions: NS, WE, NE-SW, NW-SE, and values are angles: 90, 0, 45, 135

    method : str, default = triangular
        See the ``method`` parameter.

    directional_variograms : Dict
        Dictionary with five variograms:

        * ``ISO``: isotropic,
        * ``NS``: North-South axis,
        * ``WE``: West-East axis,
        * ``NE-SW``: Northeastern-Southwestern axis,
        * ``NW-SE``: Northwestern-Southeastern axis.

    Methods
    -------
    get()
        Returns copy of calculated directional variograms or a single variogram in a specific direction.

    show()
        Plot all variograms on a single plot.
    """

    def __init__(self, input_array: np.array, step_size: float, max_range: float, weights=None, tolerance: float = 0.2,
                 method='t'):

        self.ds = input_array
        self.step_size = step_size
        self.max_range = max_range
        self.tolerance = tolerance
        self.weights = weights
        self.method = method
        self.possible_variograms = ['ISO', 'NS', 'WE', 'NE-SW', 'NW-SE']

        self.directions = {
            'NS': 90,
            'WE': 0,
            'NE-SW': 45,
            'NW-SE': 135
        }

        self.directional_variograms = {}

        self._build_experimental_variograms()

    def _build_experimental_variograms(self):

        isotropic = build_experimental_variogram(self.ds, self.step_size, self.max_range, weights=self.weights)
        self.directional_variograms['ISO'] = isotropic

        for idx, val in self.directions.items():
            variogram = build_experimental_variogram(self.ds,
                                                     self.step_size,
                                                     self.max_range,
                                                     weights=self.weights,
                                                     direction=val,
                                                     tolerance=self.tolerance,
                                                     method=self.method)
            self.directional_variograms[idx] = variogram

    def get(self, direction=None) -> Union[Dict, Type["ExperimentalVariogram"]]:
        """
        Method returns all variograms or a single variogram at a specific direction.

        Parameters
        ----------
        direction : str, default = None
            The direction of variogram from a list of ``possible_variograms`` attribute: "ISO", "NS", "WE", "NE-SW",
            "NW-SE".

        Returns
        -------
        : Union[Dict, Type[ExperimentalVariogram]]
            The dictionary with variograms for all possible directions, or a single variogram for a specific direction.
        """
        if direction is None:
            return self.directional_variograms.copy()
        else:
            if direction in self.possible_variograms:
                return self.directional_variograms[direction]

            msg = f'Given direction is not possible to retrieve, pass one direction from a possible_variograms: ' \
                  f'{self.possible_variograms} or leave None to get a dictionary with all possible variograms.'
            raise KeyError(msg)

    def show(self):
        if self.directional_variograms:
            _lags = self.directional_variograms['ISO'].lags
            _ns = self.directional_variograms['NS'].experimental_semivariances
            _we = self.directional_variograms['WE'].experimental_semivariances
            _nw_se = self.directional_variograms['NW-SE'].experimental_semivariances
            _ne_sw = self.directional_variograms['NE-SW'].experimental_semivariances
            _iso = self.directional_variograms['ISO'].experimental_semivariances

            plt.figure(figsize=(20, 8))
            plt.plot(_lags, _iso, color='#1b9e77')
            plt.plot(_lags, _ns, '--', color='#d95f02')
            plt.plot(_lags, _we, '--', color='#7570b3')
            plt.plot(_lags, _nw_se, '--', color='#e7298a')
            plt.plot(_lags, _ne_sw, '--', color='#66a61e')
            plt.title('Comparison of experimental semivariance models')
            plt.legend(['Isotropic',
                        'NS',
                        'WE',
                        'NW-SE',
                        'NE-SW'])
            plt.xlabel('Distance')
            plt.ylabel('Semivariance')
            plt.show()


class ExperimentalVariogram:
    """
    Class calculates Experimental Semivariogram and Experimental Covariogram of a given dataset.

    Parameters
    ----------
    input_array : numpy array, list, tuple
        * As a ``list`` and ``numpy array``: coordinates and their values: ``(pt x, pt y, value)``,
        * as a ``dict``: ``polyset = {'points': numpy array with coordinates and their values}``,
        * as a ``Blocks``: ``Blocks.polyset['points']``.

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

    is_semivariance : bool, optional, default=True
        Should semivariance be calculated?

    is_covariance : bool, optional, default=True
        Should covariance be calculated?

    is_variance : bool, optional, default=True
        Should variance be calculated?

    Attributes
    ----------
    input_array : numpy array
        The array with coordinates and observed values.

    experimental_semivariance_array : numpy array or None, optional, default=None
        The array of semivariance per lag in the form: ``(lag, semivariance, number of points within lag)``.

    experimental_covariance_array : numpy array or None, optional, default=None
        The array of covariance per lag in the form: ``(lag, covariance, number of points within lag)``.

    experimental_semivariances : numpy array or None, optional, default=None
        The array of semivariances.

    experimental_covariances : numpy array or None, optional, default=None
        The array of covariances.

    variance_covariances_diff : numpy array or None, optional, default=None
        The array of differences $c(0) - c(h)$.

    lags : numpy array or None, default=None
        The array of lags (upper bound for each lag).

    points_per_lag : numpy array or None, default=None
        A number of points in each lag-bin.

    variance : float or None, optional, default=None
        The variance of a dataset, if data is second-order stationary then we are able to retrieve a semivariance
        s a difference between the variance and the experimental covariance:

        .. math::

            g(h) = c(0) - c(h)

        where:

        * :math:`g(h)`: semivariance at a given lag h,
        * :math:`c(0)`: variance of a dataset,
        * :math:`c(h)`: covariance of a dataset.

        **Important!** Have in mind that it works only if process is second-order stationary (variance is the same
        for each distance bin) and if the semivariogram has the upper bound.

    step : float
        Derived from the ``step_size`` parameter.

    mx_rng : float
        Derived from the ``max_range`` parameter.

    weights : numpy array or None
        Derived from the ``weights`` parameter.

    direct: float
        Derived from the ``direction`` parameter.

    tol : float
        Derived from the ``tolerance`` parameter.

    method : str
        See the ``method`` parameter.

    Methods
    -------
    plot()
        Shows experimental variances.

    See Also
    --------
    calculate_covariance : function to calculate experimental covariance and variance of a given set of points.
    calculate_semivariance : function to calculate experimental semivariance from a given set of points.

    Examples
    --------
    >>> import numpy as np
    >>> REFERENCE_INPUT = np.array([
    ...    [0, 0, 8],
    ...    [1, 0, 6],
    ...    [2, 0, 4],
    ...    [3, 0, 3],
    ...    [4, 0, 6],
    ...    [5, 0, 5],
    ...    [6, 0, 7],
    ...    [7, 0, 2],
    ...    [8, 0, 8],
    ...    [9, 0, 9],
    ...    [10, 0, 5],
    ...    [11, 0, 6],
    ...    [12, 0, 3]
    ...    ])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = ExperimentalVariogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+--------------------+
    | lag |    semivariance    |      covariance     |    var_cov_diff    |
    +-----+--------------------+---------------------+--------------------+
    | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
    | 2.0 | 5.2272727272727275 | -0.7954545454545454 | 5.0439752555137165 |
    | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
    +-----+--------------------+---------------------+--------------------+
    """

    def __init__(self,
                 input_array: Union[np.ndarray, list, tuple],
                 step_size: float,
                 max_range: float,
                 weights=None,
                 direction: float = None,
                 tolerance: float = 1.0,
                 method='t',
                 is_semivariance=True,
                 is_covariance=True,
                 is_variance=True):

        self.input_array = None  # core structure

        if isinstance(input_array, np.ndarray):
            self.input_array = input_array
        else:
            self.input_array = np.array(input_array)

        # Object main attributes
        self.experimental_semivariance_array = None
        self.experimental_covariance_array = None
        self.lags = None
        self.experimental_semivariances = None
        self.experimental_covariances = None
        self.variance_covariances_diff = None
        self.points_per_lag = None
        self.variance = 0

        self.step = step_size
        self.mx_rng = max_range
        self.weights = weights
        self.direct = direction
        self.tol = tolerance
        self.method = method

        self.__c_sem = is_semivariance
        self.__c_cov = is_covariance
        self.__c_var = is_variance

        if is_semivariance:
            self._calculate_semivariance()
            self.lags = self.experimental_semivariance_array[:, 0]
            self.points_per_lag = self.experimental_semivariance_array[:, 2]
            self.experimental_semivariances = self.experimental_semivariance_array[:, 1]

        if is_covariance:
            self._calculate_covariance(is_variance)
            self.experimental_covariances = self.experimental_covariance_array[:, 1]

            if not is_semivariance:
                self.lags = self.experimental_covariance_array[:, 0]
                self.points_per_lag = self.experimental_covariance_array[:, 2]

            if is_variance:
                self.variance_covariances_diff = self.variance - self.experimental_covariances

    def plot(self, plot_semivariance=True, plot_covariance=False, plot_variance=False) -> None:
        """

        Parameters
        ----------
        plot_semivariance : bool, default=True
            Show semivariance on a plot. If class attribute ``is_semivariance`` is set to ``False`` then semivariance is
            not plotted and warning is printed.

        plot_covariance : bool, default=True
            Show covariance on a plot. If class attribute ``is_covariance`` is set to ``False`` then covariance
            is not plotted and warning is printed.

        plot_variance : bool, default=True
            Show variance level on a plot. If class attribute ``is_variance`` is set to ``False`` then variance is
            not plotted and warning is printed.

        Warns
        -----
        AttributeSetToFalseWarning
            Warning invoked when plotting parameter for semivariance, covariance or variance is set to ``True`` but
            class attributes to calculate those indices are set to ``False``.
        """

        # Validate parameters
        validate_plot_attributes_for_experimental_variogram_class(is_semivar=self.__c_sem,
                                                                  is_covar=self.__c_cov,
                                                                  is_var=self.__c_var,
                                                                  plot_semivar=plot_semivariance,
                                                                  plot_covar=plot_covariance,
                                                                  plot_var=plot_variance)

        # Plot
        # Cmap - 3 class Set2 https://colorbrewer2.org/#type=qualitative&scheme=Set2&n=3
        # Colorblind friendly
        # Print friendly

        legend = []
        plt.figure(figsize=(12, 6))
        if plot_semivariance and self.__c_sem:
            plt.scatter(self.lags, self.experimental_semivariances, marker='8', c='#66c2a5')
            legend.append('Experimental Semivariances')
        if plot_covariance and self.__c_cov:
            plt.scatter(self.lags, self.experimental_covariances, marker='+', c='#8da0cb')
            legend.append('Experimental Covariances')
        if plot_variance and self.__c_var:
            var_line = [self.variance for _ in self.lags]
            plt.plot(self.lags, var_line, '--', color='#fc8d62')
            legend.append('Variance')
        plt.legend(legend)
        plt.xlabel('Distance')
        plt.ylabel('Variance')
        plt.show()

    def _calculate_covariance(self, get_variance=False):
        """
        Method calculates covariance and variance.

        See : calculate_covariance function.
        """
        self.experimental_covariance_array, self.variance = calculate_covariance(
            points=self.input_array,
            step_size=self.step,
            max_range=self.mx_rng,
            direction=self.direct,
            tolerance=self.tol,
            get_c0=get_variance
        )

    def _calculate_semivariance(self):
        """
        Method calculates semivariance.

        See: calculate_semivariance function.
        """
        self.experimental_semivariance_array = calculate_semivariance(
            points=self.input_array,
            step_size=self.step,
            max_range=self.mx_rng,
            weights=self.weights,
            direction=self.direct,
            tolerance=self.tol,
            method=self.method
        )

    def __repr__(self):
        cname = 'ExperimentalVariogram'
        input_params = f'input_array={self.input_array.tolist()}, step_size={self.step}, max_range={self.mx_rng}, ' \
                       f'weights={self.weights}, direction={self.direct}, tolerance={self.tol}, ' \
                       f'is_semivariance={self.__c_sem}, is_covariance={self.__c_cov}, is_variance={self.__c_var}'
        repr_val = cname + '(' + input_params + ')'
        return repr_val

    def __str__(self):

        pretty_table = PrettyTable()

        pretty_table.field_names = ["lag", "semivariance", "covariance", "var_cov_diff"]

        if not self.__c_sem and not self.__c_cov:
            return self.__str_empty()
        else:
            if self.__c_sem and self.__c_cov:
                pretty_table.add_rows(self.__str_populate_both())
            else:
                pretty_table.add_rows(self.__str_populate_single())
            return pretty_table.get_string()

    def __str_empty(self):
        if not self.__c_var:
            return "Empty object"
        else:
            return f"Variance: {self.variance:.4f}"

    def __str_populate_both(self):
        rows = []
        if self.__c_var:
            for idx, row in enumerate(self.experimental_semivariances):
                lag = self.lags[idx]
                smv = row
                cov = self.experimental_covariances[idx]
                var_cov_diff = self.variance_covariances_diff[idx]
                rows.append([lag, smv, cov, var_cov_diff])
        else:
            for idx, row in enumerate(self.experimental_semivariances):
                lag = self.lags[idx]
                smv = row
                cov = self.experimental_covariances[idx]
                rows.append([lag, smv, cov, nan])
        return rows

    def __str_populate_single(self):
        rows = []
        if self.__c_cov:
            if self.__c_var:
                for idx, row in enumerate(self.experimental_covariances):
                    lag = self.lags[idx]
                    cov = row
                    var_cov_diff = self.variance_covariances_diff[idx]
                    rows.append([lag, nan, cov, var_cov_diff])
            else:
                for idx, row in enumerate(self.experimental_covariances):
                    lag = self.lags[idx]
                    cov = row
                    rows.append([lag, nan, cov, nan])
        else:
            for idx, row in enumerate(self.experimental_semivariances):
                lag = self.lags[idx]
                sem = row
                rows.append([lag, sem, nan, nan])
        return rows


def build_experimental_variogram(input_array: np.array,
                                 step_size: float,
                                 max_range: float,
                                 weights: np.array = None,
                                 direction: float = None,
                                 tolerance: float = 1,
                                 method='t') -> ExperimentalVariogram:
    """
    Function prepares:
      - experimental semivariogram,
      - experimental covariogram,
      - variance.

    Parameters
    ----------
    input_array : numpy array
        Spatial coordinates and their values: ``[pt x, pt y, value]`` or ``[shapely.Point(), value]``.

    step_size : float
        The distance between lags within each points are included in the calculations.

    max_range : float
        The maximum range of analysis.

    weights : numpy array or None, optional, default=None
        Weights assigned to points, index of weight must be the same as index of point.

    direction : float (in range [0, 360]), default = None
        Direction of semivariogram, values from 0 to 360 degrees:
        
        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional, default=1
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y axis and direction parameter. If ``tolerance`` is ``> 0``
        then the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance:

        * the major axis size == ``step_size``,
        * the minor axis size is ``tolerance * step_size``,
        * the baseline point is at a center of the ellipse,
        * the ``tolerance == 1`` creates an omnidirectional semivariogram.

    method : str, default = triangular
        The method used for neighbors selection. Available methods:

        * "triangle" or "t", default method where a point neighbors are selected from a triangular area,
        * "ellipse" or "e", the most accurate method but also the slowest one.

    Returns
    -------
    semivariogram_stats : EmpiricalSemivariogram
        The class with empirical semivariogram, empirical covariogram and a variance.

    See Also
    --------
    calculate_covariance : function to calculate experimental covariance and variance of a given set of points.
    calculate_semivariance : function to calculate experimental semivariance from a given set of points.
    EmpiricalSemivariogram : class that calculates and stores experimental semivariance, covariance and variance.

    Notes
    -----
    Function is an alias for ``EmpiricalSemivariogram`` class and it forces calculations of all spatial statistics
    from a given dataset.

    Examples
    --------
    >>> import numpy as np
    >>> REFERENCE_INPUT = np.array([
    ...    [0, 0, 8],
    ...    [1, 0, 6],
    ...    [2, 0, 4],
    ...    [3, 0, 3],
    ...    [4, 0, 6],
    ...    [5, 0, 5],
    ...    [6, 0, 7],
    ...    [7, 0, 2],
    ...    [8, 0, 8],
    ...    [9, 0, 9],
    ...    [10, 0, 5],
    ...    [11, 0, 6],
    ...    [12, 0, 3]
    ...    ])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = build_experimental_variogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+--------------------+
    | lag |    semivariance    |      covariance     |    var_cov_diff    |
    +-----+--------------------+---------------------+--------------------+
    | 1.0 |       4.625        |       -0.543        |       4.792        |
    | 2.0 |       5.227        |       -0.795        |       5.043        |
    | 3.0 |        6.0         |       -1.26         |       5.509        |
    +-----+--------------------+---------------------+--------------------+
    """
    semivariogram_stats = ExperimentalVariogram(
        input_array=input_array,
        step_size=step_size,
        max_range=max_range,
        weights=weights,
        direction=direction,
        tolerance=tolerance,
        method=method,
        is_semivariance=True,
        is_covariance=True,
        is_variance=True
    )
    return semivariogram_stats
