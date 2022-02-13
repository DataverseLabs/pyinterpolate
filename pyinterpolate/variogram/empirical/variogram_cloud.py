import numpy as np
from scipy.stats.stats import skew, kurtosis
from prettytable import PrettyTable

import matplotlib.pyplot as plt

from pyinterpolate.variogram.empirical.cloud import get_variogram_point_cloud


class VariogramCloud:
    """
    Class calculates Variogram Point Cloud and presents it in a scatterplot, boxplot and violinplot.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value) or (Point(), value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    direction : float (in range [0, 360]), optional, default=0
                direction of semivariogram, values from 0 to 360 degrees:
                * 0 or 180: is NS direction,
                * 90 or 270 is EW direction,
                * 45 or 225 is NE-SW direction,
                * 135 or 315 is NW-SE direction.

    tolerance : float (in range [0, 1]), optional, default=1
                If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                for 0 tolerance.
                * The minor axis size is (tolerance * step_size)
                * The major axis size is ((1 - tolerance) * step_size)
                * The baseline point is at a center of the ellipse.
                Tolerance == 1 creates an omnidirectional semivariogram.

    Attributes
    ----------
    input_array : numpy array
                  The array with coordinates and observed values.

    experimental_point_cloud : dict or None, default=None
                               Dict with lag: variances
                               {lag: [variances]}

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

    Methods
    -------
    describe()
        calculates statistics for point clouds. It is invoked by default by class __str__() method.

    plot(kind='scatter')
        plots scatterplot, boxplot or violinplot of a point cloud.

    __str__()
        prints basic info about the class parameters and calculates statistics for each lag.

    __repr__()
        reproduces class initialization with an input data.

    See Also
    --------
    get_variogram_point_cloud : function to calculate variogram point cloud, class VariogramCloud is a wrapper
                                around it.
    EmpiricalVariogram : class that calculates experimental semivariogram, experimental covariogram and data variance.

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

    def __init__(self, input_array, step_size: float, max_range: float, direction=0, tolerance=1):

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        self.input_array = input_array
        self.experimental_point_cloud = None
        self.lags = None
        self.points_per_lag = None

        self.step = step_size
        self.mx_rng = max_range
        self.direct = direction
        self.tol = tolerance

        # Calculate pt cloud
        self._calculate_point_cloud()

        # Addtional params
        self.fnames = ['lag', 'count', 'avg semivariance', 'std', 'min', '25%', 'median', '75%', 'max', 'skewness',
                       'kurtosis']

    def describe(self) -> dict:
        """
        Method calculates basic statistcs of a data: count (point pairs number), average semivariance,
            standard deviation, minimum, 1st quartile, median, 3rd quartile, maximum, skewness, kurtosis.

        Returns
        -------
        statistics : dict
                     statistics = {
                         lag_number:
                         {
                             'count': point pairs count,
                             'avg semivariance': mean semivariance,
                             'std': standard deviation,
                             'min': minimal variance,
                             '25%': first quartile of variances,
                             'median': second quartile of variances,
                             '75%': third quartile of variances,
                             'max': max variance,
                             'skewness': skewness,
                             'kurtosis': kurtosis
                         }
                     }

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

    def _calculate_point_cloud(self):
        self.experimental_point_cloud = get_variogram_point_cloud(input_array=self.input_array,
                                                                  step_size=self.step,
                                                                  max_range=self.mx_rng,
                                                                  direction=self.direct,
                                                                  tolerance=self.tol)
        self.lags = np.array(sorted(self.experimental_point_cloud.keys()))
        self.points_per_lag = []
        for lag in self.lags:
            length = len(self.experimental_point_cloud[lag])
            self.points_per_lag.append(length)

    def _box_plot(self):
        title_box = 'Boxplot of Variogram Point Cloud per lag.'
        self.__dist_plots(title_box, 'box')

    def _scatter_plot(self):
        plot_data = self.__prep_scatterplot_data()
        plt.figure(figsize=(14, 8))
        plt.scatter(x=plot_data[:, 0], y=plot_data[:, 1])
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
            ax.boxplot(plot_data_values)
        elif kind == 'violin':
            vplot = ax.violinplot(plot_data_values, showmeans=True, showmedians=True, showextrema=True)
            vplot['cmeans'].set_color('orange')
            vplot['cmedians'].set(color='black', ls='dashed')
            vplot['cmins'].set(color='black')
            vplot['cmaxes'].set(color='black')
            ax.legend([vplot['cmeans'],
                       vplot['cmedians'],
                       vplot['cmins'],
                       vplot['cmaxes']], ['mean', 'median', 'min & max'], loc='upper left')


        ax.xaxis.set_ticks(plabels)
        no_of_digits = max([len(str(x)) for x in plabels])

        if 3 < no_of_digits < 6:
            ax.tick_params(axis='x', labelrotation=30)
        elif 6 <= no_of_digits < 10:
            ax.tick_params(axis='x', labelrotation=45)
        elif no_of_digits >= 10:
            ax.tick_params(axis='x', labelrotation=90)

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

    def __prep_distplot_data(self) -> tuple:
        data = [x for x in self.experimental_point_cloud.values()]
        x_tick_labels = list(self.experimental_point_cloud.keys())
        return data, x_tick_labels

    def __prep_scatterplot_data(self) -> np.array:
        data = []
        for lag in self.lags:
            data.extend([[lag, x] for x in self.experimental_point_cloud[lag]])
        data = np.array(data)
        return data

    @staticmethod
    def __str_empty():
        return 'Empty Object'


if __name__ == '__main__':
    from pyinterpolate.test.test_variogram.empirical.consts import get_armstrong_data

    data = get_armstrong_data()
    cloud = VariogramCloud(data, 1, 8)
    cloud.plot(kind='violin')
