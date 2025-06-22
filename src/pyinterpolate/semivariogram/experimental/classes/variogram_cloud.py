import copy
from typing import Collection, Dict, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import skew, kurtosis

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import (
    ExperimentalVariogram)
from pyinterpolate.transform.statistical import remove_outliers


def build_swarmplot_input(data: Dict, bins=None):
    """
    Function prepares data for lagged beeswarm plot.

    Parameters
    ----------
    data : Dict
        ``{lag: [values]}``

    bins : int, optional
        If ``None`` given then number of bins per lag is chosen automatically.

    Returns
    -------
    : numpy array
        [lags (x-coordinates), values]
    """

    xs_arr = []
    ys_arr = []

    lags = list(data.keys())
    step_sizes = [x - lags[idx] for idx, x in enumerate(lags[1:])]
    ss = [lags[0]].copy()
    ss.extend(step_sizes)

    for idx, dict_obj in enumerate(data.items()):

        center = dict_obj[0]
        values = dict_obj[1]
        step_size = ss[idx]

        # bin values
        # TODO: set max points per level
        if bins is None:
            histogram, bin_edges = np.histogram(values, bins='auto')
        else:
            histogram, bin_edges = np.histogram(values, bins=bins)

        # Now prepare x-coordinates per lag
        x_indexes = []
        y_values = []

        # Define limits
        max_no = np.max(histogram)

        limits = [
            step_size * _get_bin_width(x, max_no) for x in histogram
        ]

        lower_limits = [center - l for l in limits]
        upper_limits = [center + l for l in limits]

        for jdx, no_points in enumerate(histogram):
            xc = np.linspace(lower_limits[jdx],
                             upper_limits[jdx],
                             no_points)
            yc = [bin_edges[jdx+1] for _ in xc]
            x_indexes.extend(xc)
            y_values.extend(yc)

        xs_arr.extend(x_indexes)
        ys_arr.extend(y_values)

    arr = np.array([xs_arr, ys_arr]).transpose()
    return arr


class VariogramCloud:
    """
    Class calculates Variogram Point Cloud and presents it in a scatterplot,
    boxplot or violinplot.

    Parameters
    ----------
    ds : numpy array
        ``[x, y, value]``

    step_size : float
        The fixed distance between lags grouping point neighbors.

    max_range : float
        The maximum distance at which the semivariance is calculated.

    direction : float, optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y-axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    dir_neighbors_selection_method : str, default = 't'
        Neighbors selection in within a given angle. Available methods:

        * "triangle" or "t", default method, where point neighbors are
          selected from a triangular area,
        * "ellipse" or "e", the more accurate method but also the slowest.

    custom_bins : numpy array, optional
        Custom bins for semivariance calculation. If provided, then parameter
        ``step_size`` is ignored and ``max_range`` is set to the final bin
        distance.

    custom_weights : numpy array, optional
        Custom weights assigned to points. Only semivariance values are
        weighted.

    Attributes
    ----------
    semivariances : Dict
        Lag - all semivariances between point pairs:
        ``{lag: [semivariances], }``.

    lags : ArrayLike
        Lags.

    direction : float
        See ``direction`` parameter.

    tolerance : float
        See ``tolerance`` parameter.

    Methods
    -------
    average_semivariance()
        Returns ``lag, average semivariance, number of point pairs``
        as a numpy array.

    describe()
        The point cloud statistics.

    plot()
        Plots scatterplot, boxplot or violinplot of the point cloud.

    remove_outliers()
        Removes outliers from the semivariance plots.

    See Also
    --------
    ExperimentalVariogram : class that calculates experimental semivariogram,
        experimental covariogram and data variance.
    """

    def __init__(self,
                 ds: np.ndarray,
                 step_size: float = None,
                 max_range: float = None,
                 direction: float = None,
                 tolerance: float = None,
                 dir_neighbors_selection_method: str = 't',
                 custom_bins: Union[np.ndarray, Collection] = None,
                 custom_weights: np.ndarray = None):

        self._experimental_variogram = ExperimentalVariogram(
            ds=ds,
            step_size=step_size,
            max_range=max_range,
            direction=direction,
            tolerance=tolerance,
            dir_neighbors_selection_method=dir_neighbors_selection_method,
            custom_bins=custom_bins,
            custom_weights=custom_weights,
            is_semivariance=True,
            is_covariance=False,
            as_cloud=True
        )

        self.semivariances = self._experimental_variogram.point_cloud_semivariances
        self.lags = self._experimental_variogram.lags
        self.direction = direction
        self.tolerance = tolerance

        # Additional params
        self._stats_names = ['lag',
                             'count',
                             'mean',
                             'std',
                             'min',
                             '25%',
                             'median',
                             '75%',
                             'max',
                             'skewness',
                             'kurtosis']

    def average_semivariance(self):
        """
        Returns mean of semivariances for each lag - which is equal to
        the experimental semivariogram output.

        Returns
        -------
        : numpy array
            Mean of semivariances for each lag.
        """
        ds = []
        for l in self.lags:
            ds.append(
                [
                    l,
                    np.mean(self.semivariances[l]),
                    len(self.semivariances[l])
                ]
            )
        return np.array(ds)

    def describe(self, as_dataframe=False) -> Union[Dict, pd.DataFrame]:
        """
        Method calculates basic statistics. Includes count (point pairs
        number), average semivariance, standard deviation, minimum,
        1st quartile, median, 3rd quartile, maximum, skewness, and kurtosis.

        Returns
        -------
        statistics : Dict
            >>> statistics = {
            ...      lag_number:
            ...      {
            ...          'count': point pairs count,
            ...          'mean': mean semivariance,
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

        # Get stats per lag
        statistics = {}
        for lag in self.lags:
            lag_dict = {}
            data = self.semivariances[lag]
            pairs_count = len(data)
            lag_dict['count'] = pairs_count
            avg_semi = np.mean(data) / 2
            lag_dict['mean'] = avg_semi
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

        if as_dataframe:
            return pd.DataFrame(statistics)
        else:
            return statistics

    def experimental_semivariances(self) -> ExperimentalVariogram:
        """
        Returns experimental semivariogram.

        Returns
        -------
        : ExperimentalVariogram
            Experimental semivariogram object.
        """
        return self._experimental_variogram

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

        if kind == 'scatter':
            self._scatter_plot()
        elif kind == 'box':
            self._box_plot()
        elif kind == 'violin':
            self._violin_plot()
        else:
            msg = (f'Plot kind {kind} is not available. '
                   f'Use "scatter", "box" or "violin" instead.')
            raise KeyError(msg)
        return True

    def remove_outliers(self,
                        method='zscore',
                        z_lower_limit=-3,
                        z_upper_limit=3,
                        iqr_lower_limit=1.5,
                        iqr_upper_limit=1.5,
                        inplace=False):
        """
        Method cleans semivariogram point cloud from outliers.

        Parameters
        ----------
        method : str, default='zscore'
            Method used to detect outliers. Can be 'zscore' or 'iqr'.

        z_lower_limit : float
            Number of standard deviations from the mean to the left side
            of a distribution. Must be lower than 0.

        z_upper_limit : float
            Number of standard deviations from the mean to the right side
            of a distribution. Must be greater than 0.

        iqr_lower_limit : float
            Number of standard deviations from the 1st quartile into
            the lowest values. Must be greater or equal to zero.

        iqr_upper_limit : float
            Number of standard deviations from the 3rd quartile into
            the largest values. Must be greater or equal to zero.

        inplace : bool, default = False
            Overwrite semivariances or return new object.

        Returns
        -------
        : VariogramCloud
            If ``inplace`` is set to ``False`` then method returns new
            instance of an object with cleaned semivariances.

        Raises
        ------
        RunetimeError
            The attribute experimental_point_cloud is not calculated.
        """
        if inplace:
            self.semivariances = remove_outliers(
                self.semivariances,
                method=method,
                z_lower_limit=z_lower_limit,
                z_upper_limit=z_upper_limit,
                iqr_lower_limit=iqr_lower_limit,
                iqr_upper_limit=iqr_upper_limit)
        else:
            new_instance = copy.deepcopy(self)
            new_instance.semivariances = remove_outliers(
                self.semivariances,
                method=method,
                z_lower_limit=z_lower_limit,
                z_upper_limit=z_upper_limit,
                iqr_lower_limit=iqr_lower_limit,
                iqr_upper_limit=iqr_upper_limit
            )
            return new_instance

    def _box_plot(self):
        title_box = 'Boxplot of Variogram Point Cloud per lag.'
        self.__dist_plots(title_box, 'box')

    def _scatter_plot(self):
        import matplotlib.pyplot as plt

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

    def __str__(self):
        pretty_table = PrettyTable()
        pretty_table.field_names = self._stats_names

        pretty_table.add_rows(self.__desc_str())
        table = '\n\n' + pretty_table.get_string()
        return table

    def __dist_plots(self, title_label: str, kind='box'):
        import matplotlib.pyplot as plt

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
                       vplot['cmaxes']], ['mean', 'median', 'min & max'],
                      loc='upper left')

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
            for fname in self._stats_names:
                row.append(vals[fname])
            rows.append(row)
        return rows

    def __prep_distplot_data(self):
        data = list(self.semivariances.values())
        x_tick_labels = list(self.semivariances.keys())
        return data, x_tick_labels

    def __prep_scatterplot_data(self) -> np.array:

        ds = build_swarmplot_input(data=self.semivariances)

        return ds


def _get_bin_width(no, max_no, max_width=0.3):
    """
    Function gets bin width on a plot.

    Parameters
    ----------
    no : int
        The number of points within bin.

    max_no : int
        The maximum number of points within all bins.

    max_width : float
        The maximum deviation from the lag center in percent.

    Returns
    -------
    : float
        The deviation from the lag center.
    """

    bin_width = (no * max_width) / max_no
    return bin_width
