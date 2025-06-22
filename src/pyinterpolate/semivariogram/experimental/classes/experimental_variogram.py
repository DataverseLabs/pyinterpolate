from typing import Union, Collection

import numpy as np
from prettytable import PrettyTable

from pyinterpolate.semivariogram.experimental.experimental_covariogram import calculate_covariance
from pyinterpolate.core.data_models.experimental_variogram import \
    ExperimentalVariogramModel
from pyinterpolate.core.data_models.points import VariogramPoints
from pyinterpolate.core.validators.experimental_semivariance import \
    validate_plot_attributes_for_experimental_variogram
from pyinterpolate.semivariogram.experimental.experimental_semivariogram import \
    point_cloud_semivariance, calculate_semivariance


class ExperimentalVariogram:
    """
    Class calculates Experimental Semivariogram and Experimental
    Covariogram of a given dataset.

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

        * 0 or 180: is E-W,
        * 90 or 270 is N-S,
        * 45 or 225 is NE-SW,
        * 135 or 315 is NW-SE.

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

    is_semivariance : bool, default=True
        Calculate experimental semivariance.

    is_covariance : bool, default=True
        Calculate experimental coviariance.

    as_cloud : bool
        Calculate semivariance point-pairs cloud.

    Attributes
    ----------
    semivariances : numpy array
        1-D array with semivariances ordered by lags.

    Methods
    -------
    plot()
        Shows experimental variances.

    See Also
    --------
    calculate_covariance : function to calculate experimental covariance
    calculate_semivariance : function to calculate experimental semivariance

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
    >>> empirical_smv = ExperimentalVariogram(REFERENCE_INPUT,
    ...                                       step_size=STEP_SIZE,
    ...                                       max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+
    | lag |    semivariance    |      covariance     |
    +-----+--------------------+---------------------+
    | 1.0 |       4.625        | -0.5434027777777798 |
    | 2.0 | 5.2272727272727275 | -0.7954545454545454 |
    | 3.0 |        6.0         | -1.2599999999999958 |
    +-----+--------------------+---------------------+

    """

    def __init__(self,
                 ds: np.ndarray,
                 step_size: float = None,
                 max_range: float = None,
                 direction: float = None,
                 tolerance: float = None,
                 dir_neighbors_selection_method: str = 't',
                 custom_bins: Union[np.ndarray, Collection] = None,
                 custom_weights: np.ndarray = None,
                 is_semivariance=True,
                 is_covariance=True,
                 as_cloud=False):

        ds = VariogramPoints(ds)

        self.ds = ds.points  # core structure
        # Object main attributes
        if custom_bins is None:
            self.lags = None
        else:
            self.lags = custom_bins
        self.points_per_lag = None
        self.semivariances = None
        self.covariances = None

        # point cloud
        self.point_cloud_semivariances = None

        self.variance = np.var(self.ds[:, -1])

        self.step_size = step_size
        self.max_range = max_range
        self.custom_weights = custom_weights
        self.direction = direction
        self.tolerance = tolerance
        self.method = dir_neighbors_selection_method
        self.as_cloud = as_cloud
        self.__c_sem = is_semivariance
        self.__c_cov = is_covariance

        if is_semivariance:
            self._calculate_semivariance()
        if is_covariance:
            self._calculate_covariance()
        if as_cloud:
            self._calculate_semivariance_point_cloud()

        # update base model
        self.model = self.get_model_params()

    def plot(self,
             semivariance=True,
             covariance=True,
             variance=True) -> None:
        """
        Plots semivariance, covariance, and variance.

        Parameters
        ----------
        semivariance : bool, default=True
            Show semivariance on a plot. If class attribute
            ``is_semivariance`` is set to ``False`` then semivariance is
            not plotted and warning is printed.

        covariance : bool, default=True
            Show covariance on a plot. If class attribute
            ``is_covariance`` is set to ``False`` then covariance
            is not plotted and warning is printed.

        variance : bool, default=True
            Show variance level on a plot.

        Warns
        -----
        AttributeSetToFalseWarning
            Warning invoked when plotting parameter for semivariance,
            covariance or variance is set to ``True`` but
            class attributes to calculate those indices are set to ``False``.
        """
        import matplotlib.pyplot as plt

        # Validate parameters
        validate_plot_attributes_for_experimental_variogram(
            is_semivar=self.__c_sem,
            is_covar=self.__c_cov,
            plot_semivar=semivariance,
            plot_covar=covariance)

        # Plot
        # Cmap - 3 class Set2
        # https://colorbrewer2.org/#type=qualitative&scheme=Set2&n=3
        # Colorblind friendly
        # Print friendly

        legend = []
        plt.figure(figsize=(12, 6))
        if semivariance and self.__c_sem:
            plt.scatter(self.lags, self.semivariances, marker='8',
                        c='#66c2a5')
            legend.append('Experimental Semivariances')
        if covariance and self.__c_cov:
            plt.scatter(self.lags, self.covariances, marker='+',
                        c='#8da0cb')
            legend.append('Experimental Covariances')
        if variance:
            var_line = [self.variance for _ in self.lags]
            plt.plot(self.lags, var_line, '--', color='#fc8d62')
            legend.append('Variance')
        plt.legend(legend)
        plt.xlabel('Distance')
        plt.ylabel('Variance')
        plt.show()

    def _calculate_covariance(self):
        """
        Method calculates covariance and variance.

        See : ``calculate_covariance`` function.
        """
        experimental_covariance_array = calculate_covariance(
            ds=self.ds,
            step_size=self.step_size,
            max_range=self.max_range,
            direction=self.direction,
            tolerance=self.tolerance,
            dir_neighbors_selection_method=self.method,
            custom_bins=self.lags
        )

        if self.lags is None:
            self.lags = experimental_covariance_array[:, 0]

        if self.points_per_lag is None:
            self.points_per_lag = experimental_covariance_array[:, -1]

        self.covariances = experimental_covariance_array[:, 1]

    def _calculate_semivariance_point_cloud(self):
        """
        Method calculates point cloud semivariance.

        See: ``calculate_semivariance`` function.
        """

        experimental_cloud = point_cloud_semivariance(
            ds=self.ds,
            step_size=self.step_size,
            max_range=self.max_range,
            direction=self.direction,
            tolerance=self.tolerance,
            dir_neighbors_selection_method=self.method,
            custom_bins=self.lags,
            custom_weights=self.custom_weights
        )

        self.point_cloud_semivariances = experimental_cloud

    def _calculate_semivariance(self):
        """
        Method calculates semivariance.

        See: ``calculate_semivariance`` function.
        """
        experimental_semivariance_array = calculate_semivariance(
            ds=self.ds,
            step_size=self.step_size,
            max_range=self.max_range,
            direction=self.direction,
            tolerance=self.tolerance,
            dir_neighbors_selection_method=self.method,
            custom_bins=self.lags,
            custom_weights=self.custom_weights
        )

        if self.lags is None:
            self.lags = experimental_semivariance_array[:, 0]

        if self.points_per_lag is None:
            self.points_per_lag = experimental_semivariance_array[:, -1]

        self.semivariances = experimental_semivariance_array[:, 1]

    def get_model_params(self):
        model_parameters = {
            "lags": self.lags,
            "points_per_lag": self.points_per_lag,
            "semivariances": self.semivariances,
            "covariances": self.covariances,
            "variance": self.variance,
            "direction": self.direction,
            "tolerance": self.tolerance,
            "max_range": self.max_range,
            "step_size": self.step_size,
            "custom_weights": self.custom_weights
        }
        return ExperimentalVariogramModel(**model_parameters)

    def __repr__(self):
        """
        ds: np.ndarray,
        step_size: float = None,
        max_range: float = None,
        direction: float = None,
        tolerance: float = None,
        dir_neighbors_selection_method: str = 't',
        custom_bins: Union[np.ndarray, Collection] = None,
        custom_weights: np.ndarray = None,
        is_semivariance=True,
        is_covariance=True
        """

        cname = 'ExperimentalVariogram'

        input_params = (f'ds={self.ds.tolist()},'
                        f'step_size={self.step_size},'
                        f'max_range={self.max_range},'
                        f'direction={self.direction},'
                        f'tolerance={self.tolerance},'
                        f'dir_neighbors_selection_method={self.method},'
                        f'custom_bins={self.lags.tolist()},'
                        f'custom_weights={self.custom_weights.tolist()},'
                        f'is_semivariance={self.__c_sem},'
                        f'is_covariance={self.__c_cov}')

        repr_val = cname + '(' + input_params + ')'
        return repr_val

    def __str__(self):

        pretty_table = PrettyTable()

        pretty_table.field_names = ["lag",
                                    "semivariance",
                                    "covariance"]

        if not self.__c_sem and not self.__c_cov:
            return self.__str_empty()
        else:
            if self.__c_sem and self.__c_cov:
                pretty_table.add_rows(self.__str_populate_both())
            else:
                pretty_table.add_rows(self.__str_populate_single())
            return pretty_table.get_string()

    def __str_empty(self):
        return (f"Variance: {self.variance:.4f}. "
                f"Other parameters not calculated yet.")

    def __str_populate_both(self):
        rows = []
        for idx, row in enumerate(self.semivariances):
            lag = self.lags[idx]
            smv = row
            cov = self.covariances[idx]
            rows.append([lag, smv, cov])

        return rows

    def __str_populate_single(self):
        rows = []
        if self.__c_cov:
            for idx, row in enumerate(self.covariances):
                lag = self.lags[idx]
                cov = row
                rows.append([lag, np.nan, cov])
        else:
            for idx, row in enumerate(self.semivariances):
                lag = self.lags[idx]
                sem = row
                rows.append([lag, sem, np.nan])
        return rows


def build_experimental_variogram(ds: np.ndarray,
                                 step_size: float = None,
                                 max_range: float = None,
                                 direction: float = None,
                                 tolerance: float = None,
                                 dir_neighbors_selection_method: str = 't',
                                 custom_bins: Union[
                                     np.ndarray, Collection
                                 ] = None,
                                 custom_weights: np.ndarray = None,
                                 is_semivariance=True,
                                 is_covariance=True,
                                 as_cloud=False):
    """
    Function is an alias to ``ExperimentalVariogram()``.

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

    is_semivariance : bool, default=True
        Calculate experimental semivariance.

    is_covariance : bool, default=True
        Calculate experimental coviariance.

    as_cloud : bool
        Calculate semivariance point-pairs cloud.

    Returns
    -------
    : ExperimentalVariogram
    """

    exp_var = ExperimentalVariogram(
        ds=ds,
        step_size=step_size,
        max_range=max_range,
        direction=direction,
        tolerance=tolerance,
        dir_neighbors_selection_method=dir_neighbors_selection_method,
        custom_bins=custom_bins,
        custom_weights=custom_weights,
        is_semivariance=is_semivariance,
        is_covariance=is_covariance,
        as_cloud=as_cloud
    )
    return exp_var
