"""
Cloud variogram class and functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict

import numpy as np
from collections import OrderedDict

from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.stats.stats import skew, kurtosis
from shapely.geometry import Point

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.processing.select_values import select_points_within_ellipse, select_values_in_range
from pyinterpolate.processing.transform.statistics import remove_outliers
from pyinterpolate.variogram.utils.exceptions import validate_direction, validate_points, validate_tolerance
from pyinterpolate.variogram.utils.plots import build_swarmplot_input


def omnidirectional_point_cloud(input_array: np.array,
                                step_size: float,
                                max_range: float) -> dict:
    """
    Function calculates lagged omnidirectional point cloud.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    Returns
    -------
    variogram_cloud : dict
                      {Lag: array of semivariances within a given lag}
    """
    distances = calc_point_to_point_distance(input_array[:, :-1])
    lags = np.arange(step_size, max_range, step_size)
    variogram_cloud = OrderedDict()

    # Calculate squared differences
    # They are halved to be compatibile with semivariogram

    for h in lags:
        distances_in_range = select_values_in_range(distances, h, step_size)
        if len(distances_in_range[0]) == 0:
            if h == lags[0]:
                variogram_cloud[h] = []
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
        else:
            sems = (input_array[distances_in_range[0], 2] - input_array[distances_in_range[1], 2]) ** 2
            variogram_cloud[h] = sems
    return variogram_cloud


def directional_point_cloud(input_array: np.array,
                            step_size: float,
                            max_range: float,
                            direction: float,
                            tolerance: float) -> dict:
    """
    Function calculates lagged variogram point cloud. Variogram is calculated as a squared difference of each point
        against other point within range specified by step_size parameter.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:
        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
                Value in range (0-1] to calculate semi-minor axis length of the search area. If tolerance is close
                to 0 then points must be placed at a single line with beginning in the origin of coordinate system
                and direction given by y axis and direction parameter.
                    * The major axis length == step_size,
                    * The minor axis size == tolerance * step_size.
                    * Tolerance == 1 creates the omnidirectional semivariogram.

    Returns
    -------
    variogram_cloud : dict
                      {Lag: array of semivariances within a given lag}
    """

    variogram_cloud = OrderedDict()
    lags = np.arange(step_size, max_range, step_size)

    for h in lags:
        variogram_vars_list = []
        for point in input_array:
            coordinates = point[:-1]

            mask = select_points_within_ellipse(
                coordinates,
                input_array[:, :-1],
                h,
                step_size,
                direction,
                tolerance
            )

            points_in_range = input_array[mask, -1]

            # Calculate semivariances
            if len(points_in_range) > 0:
                svars = (points_in_range - point[-1]) ** 2
                variogram_vars_list.extend(svars)

        if len(variogram_vars_list) == 0:
            if h == lags[0]:
                variogram_cloud[h] = []
            else:
                msg = f'There are no neighbors for a lag {h}, the process has been stopped.'
                raise RuntimeError(msg)
        else:
            variogram_cloud[h] = variogram_vars_list

    return variogram_cloud


def build_variogram_point_cloud(input_array: np.array,
                                step_size: float,
                                max_range: float,
                                direction=None,
                                tolerance=1.0) -> dict:
    """
    Function calculates lagged variogram point cloud. Variogram is calculated as a squared difference of each point
        against other point within range specified by step_size parameter.

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

    direction : float (in range [0, 360]), default=None
        Direction of semivariogram, values from 0 to 360 degrees:
        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional, default=1
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y axis and direction parameter. If ``tolerance`` is ``> 0`` then
        the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    Returns
    -------
    variogram_cloud : Dict
        ``{Lag: array of semivariances within a given lag}``
    """

    # START:VALIDATION
    # Test size of points array and input data types
    validate_points(input_array)

    # Transform Point into floats
    is_point_type = isinstance(input_array[0][0], Point)
    if is_point_type:
        input_array = np.array([[x[0].x, x[0].y, x[1]] for x in input_array])

    # Test directions if provided
    validate_direction(direction)

    # Test provided tolerance parameter
    validate_tolerance(tolerance)
    # END:VALIDATION

    if direction is None:
        return omnidirectional_point_cloud(input_array, step_size, max_range)
    else:
        return directional_point_cloud(input_array, step_size, max_range, direction, tolerance)


class VariogramCloud:
    """
    Class calculates Variogram Point Cloud and presents it in a scatterplot, boxplot and violinplot.

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

    direction : float (in range [0, 360]), optional, default=None
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional, default=1
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y axis and direction parameter. If ``tolerance`` is ``> 0`` then
        the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    Attributes
    ----------
    input_array : numpy array
        The array with coordinates and observed values.

    experimental_point_cloud : dict or None, default=None
        Dict ``{lag: [variances]}``.

    lags : numpy array or None, default=None
         The array of lags (upper bound for each lag).

    points_per_lag : int or None, default=None
        A number of points in each lag-bin.

    step : float
        Derived from the step_size parameter.

    mx_rng : float
        Derived from the  max_range parameter.

    direct: float
        Derived from the direction parameter.

    tol : float
        Derived from the tolerance parameter.

    calculate_on_creation : bool, default=True
        Perform calculations of semivariogram point cloud when object is initialized.

    Methods
    -------
    calculate_experimental_variogram()
        Method calculates experimental variogram from a point cloud.

    describe()
        calculates statistics for point clouds. It is invoked by default by class __str__() method.

    plot(kind='scatter')
        plots scatterplot, boxplot or violinplot of a point cloud.

    remove_outliers()
        Removes outliers from a semivariance scatterplots.

    See Also
    --------
    get_variogram_point_cloud : function to calculate variogram point cloud, class VariogramCloud is a wrapper
                                around it.
    ExperimentalVariogram : class that calculates experimental semivariogram, experimental covariogram and data
                            variance.

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
    >>> point_cloud = VariogramCloud(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    # >>> print(point_cloud)
    # +-----+--------------------+---------------------+--------------------+
    # | lag |    count    |      avg semivariance    |    std  | min | 25% | median | 75% | max | skewness | kurtosis |
    # +-----+--------------------+---------------------+--------------------+
    # | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
    # | 2.0 | 5.22727272727272 | -0.7954545454545454 | 5.0439752555137165 |
    # | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
    # +-----+--------------------+---------------------+--------------------+
    """

    def __init__(self, input_array, step_size: float, max_range: float, direction=0, tolerance=1,
                 calculate_on_creation=True):

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        # Validate input array
        validate_points(input_array)

        self.input_array = input_array
        self.experimental_point_cloud = None
        self.lags = None
        self.points_per_lag = None
        self.experimental_variogram = None

        self.step = step_size
        self.mx_rng = max_range
        self.direct = direction
        self.tol = tolerance

        # Calculate pt cloud
        if calculate_on_creation:
            self._calculate_point_cloud()

        # Addtional params
        self.fnames = ['lag', 'count', 'avg semivariance', 'std', 'min', '25%', 'median', '75%', 'max', 'skewness',
                       'kurtosis']

    def calculate_experimental_variogram(self):
        """
        Method transforms the experimental point cloud into the experimental variogram.

        Raises
        ------
        RunetimeError
            The attribute experimental_point_cloud is not calculated.
        """
        experimental_semivariogram = []

        if self.experimental_point_cloud is None:
            raise RuntimeError('You must calculate experimental_point_cloud first before you remove outliers.')

        for _key, _values in self.experimental_point_cloud.items():
            try:
                mean_semivariance_value = np.mean(_values) / 2
                length = len(_values)
            except ZeroDivisionError:
                # There are no points for this lag
                mean_semivariance_value = 0
                length = 0
            # Check if any nan
            if np.isnan(mean_semivariance_value):
                mean_semivariance_value = 0
                length = 0

            experimental_semivariogram.append([_key, mean_semivariance_value, length])

        experimental_semivariogram = np.array(experimental_semivariogram)

        self.experimental_variogram = experimental_semivariogram
        return experimental_semivariogram

    def describe(self) -> Dict:
        """
        Method calculates basic statistcs of a data: count (point pairs number), average semivariance,
        standard deviation, minimum, 1st quartile, median, 3rd quartile, maximum, skewness, kurtosis.

        Returns
        -------
        statistics : Dict
            >>> statistics = {
            ...      lag_number:
            ...      {
            ...          'count': point pairs count,
            ...          'avg semivariance': mean semivariance,
            ...          'std': standard deviation,
            ...          'min': minimal variance,
            ...          '25%': first quartile of variances,
            ...          'median': second quartile of variances,
            ...          '75%': third quartile of variances,
            ...          'max': max variance,
            ...          'skewness': skewness,
            ...          'kurtosis': kurtosis
            ...      }
            ...  }

        """

        if self.experimental_point_cloud is None:
            msg = 'Description of empty variogram cloud is not possible!'
            raise ValueError(msg)

        # Get stats per lag
        statistics = {}
        for lag in self.lags:
            lag_dict = {}
            data = self.experimental_point_cloud[lag]
            pairs_count = len(data)
            lag_dict['count'] = pairs_count
            avg_semi = np.mean(data) / 2
            lag_dict['avg semivariance'] = avg_semi
            std_cloud = np.std(data)
            lag_dict['std'] = std_cloud
            min_cloud = np.min(data)
            lag_dict['min'] = min_cloud
            max_cloud = np.max(data)
            lag_dict['max'] = max_cloud
            first_q = np.quantile(data, q=0.25)
            lag_dict['25%'] = first_q
            med_cloud = np.median(data)
            lag_dict['median'] = med_cloud
            third_q = np.quantile(data, q=0.75)
            lag_dict['75%'] = third_q
            skew_cloud = skew(data)
            lag_dict['skewness'] = skew_cloud
            kurt_cloud = kurtosis(data)
            lag_dict['kurtosis'] = kurt_cloud
            lag_dict['lag'] = lag

            # Update statistics
            statistics[lag] = lag_dict

        return statistics

    def plot(self, kind='scatter'):
        """
        Method plots variogram point cloud.

        Parameters
        ----------
        kind : string, default='scatter'
               available plot types: 'scatter', 'box', 'violin'

        Returns
        -------
        : bool
            ``True`` if Figure was plotted.
        """

        if self.experimental_point_cloud is None:
            msg = 'Plot of empty variogram cloud is not possible!'
            raise ValueError(msg)

        if kind == 'scatter':
            self._scatter_plot()
        elif kind == 'box':
            self._box_plot()
        elif kind == 'violin':
            self._violin_plot()
        else:
            msg = f'Plot kind {kind} is not available. Use "scatter", "box" or "violin" instead.'
            raise KeyError(msg)
        return True


    def remove_outliers(self, method='zscore',
                        z_lower_limit=-3,
                        z_upper_limit=3,
                        iqr_lower_limit=1.5,
                        iqr_upper_limit=1.5,
                        inplace=False):
        """

        Parameters
        ----------
        method : str, default='zscore'
            Method used to detect outliers. Can be 'zscore' or 'iqr'.

        z_lower_limit : float
            Number of standard deviations from the mean to the left side of a distribution. Must be lower than 0.

        z_upper_limit : float
            Number of standard deviations from the mean to the right side of a distribution. Must be greater than 0.

        iqr_lower_limit : float
            Number of standard deviations from the 1st quartile into the lowest values. Must be greater or
            equal to zero.

        iqr_upper_limit : float
            Number of standard deviations from the 3rd quartile into the largest values. Must be greater or
            equal to zero.

        inplace : bool, default=False
            If set to True then method updates experimental_point_cloud attribute of the existing object and returns
            nothing. Else new VariogramCloud object is returned.

        Returns
        -------
        cleaned : VariogramCloud
            VariogramCloud object with removed outliers from the experimental_point_cloud attribute.

        Raises
        ------
        RunetimeError
            The attribute experimental_point_cloud is not calculated.
        """
        if self.experimental_point_cloud is None:
            raise RuntimeError('You must calculate experimental_point_cloud first before you remove outliers.')

        processed = remove_outliers(self.experimental_point_cloud, method=method,
                                    z_lower_limit=z_lower_limit, z_upper_limit=z_upper_limit,
                                    iqr_lower_limit=iqr_lower_limit, iqr_upper_limit=iqr_upper_limit)

        if inplace:
            self.experimental_point_cloud = processed
        else:
            vc = VariogramCloud(input_array=self.input_array.copy(),
                                step_size=self.step,
                                max_range=self.mx_rng, direction=0, tolerance=1, calculate_on_creation=False)
            vc.experimental_point_cloud = processed
            vc.lags = self.lags.copy()
            new_points_per_lag = []

            for lag in self.lags:
                length = len(processed[lag])
                new_points_per_lag.append(length)

            vc.points_per_lag = new_points_per_lag
            return vc

    def _calculate_point_cloud(self):
        self.experimental_point_cloud = build_variogram_point_cloud(input_array=self.input_array,
                                                                    step_size=self.step,
                                                                    max_range=self.mx_rng,
                                                                    direction=self.direct,
                                                                    tolerance=self.tol)
        self.lags = np.array(list(self.experimental_point_cloud.keys()))
        self.points_per_lag = []
        for lag in self.lags:
            length = len(self.experimental_point_cloud[lag])
            self.points_per_lag.append(length)

    def _box_plot(self):
        title_box = 'Boxplot of Variogram Point Cloud per lag.'
        self.__dist_plots(title_box, 'box')

    def _scatter_plot(self):

        ds = self.__prep_scatterplot_data()
        plt.figure(figsize=(14, 8))
        xs = ds[:, 0]
        ys = ds[:, 1]
        plt.scatter(x=xs,
                    y=ys,
                    s=0.2)
        plt.title('Variogram Point Cloud per lag.')
        plt.show()

    def _violin_plot(self):
        title_violin = 'Variogram Point Cloud distributions per lag.'
        self.__dist_plots(title_violin, 'violin')

    def __repr__(self):
        cname = 'VariogramCloud'
        input_params = f'input_array={self.input_array.tolist()}, step_size={self.step}, max_range={self.mx_rng}, ' \
                       f'direction={self.direct}, tolerance={self.tol}'
        repr_val = cname + '(' + input_params + ')'
        return repr_val

    def __str__(self):
        pretty_table = PrettyTable()
        pretty_table.field_names = self.fnames

        if self.experimental_point_cloud is None:
            return self.__str_empty()
        else:
            pretty_table.add_rows(self.__desc_str())
            return pretty_table.get_string()

    def __dist_plots(self, title_label: str, kind='box'):
        plot_data_values, plabels = self.__prep_distplot_data()
        fig, ax = plt.subplots(figsize=(14, 8))

        if kind == 'box':
            ax.boxplot(plot_data_values, showfliers=False)
        elif kind == 'violin':
            vplot = ax.violinplot(plot_data_values,
                                  showmeans=True,
                                  showmedians=True,
                                  showextrema=True)

            vplot['cmeans'].set_color('orange')
            vplot['cmedians'].set(color='black', ls='dashed')
            vplot['cmins'].set(color='black')
            vplot['cmaxes'].set(color='black')
            ax.legend([vplot['cmeans'],
                       vplot['cmedians'],
                       vplot['cmins'],
                       vplot['cmaxes']], ['mean', 'median', 'min & max'], loc='upper left')

        str_plabels = [str(f'{plabel:.2f}') for plabel in plabels]
        ax.set_xticks(np.arange(1, len(plabels) + 1))
        ax.set_xticklabels(str_plabels)

        plt.title(title_label)
        plt.show()

    def __desc_str(self):
        description = self.describe()
        rows = []
        for lag in self.lags:
            vals = description[lag]
            row = []
            for fname in self.fnames:
                row.append(vals[fname])
            rows.append(row)
        return rows

    def __prep_distplot_data(self):
        data = list(self.experimental_point_cloud.values())
        x_tick_labels = list(self.experimental_point_cloud.keys())
        return data, x_tick_labels

    def __prep_scatterplot_data(self) -> np.array:

        ds = build_swarmplot_input(data=self.experimental_point_cloud,
                                   step_size=self.step)

        return ds

    @staticmethod
    def __str_empty():
        return 'Empty Object'
