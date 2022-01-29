import numpy as np
from numpy import nan
from prettytable import PrettyTable
from pyinterpolate.variogram.empirical.covariance import calculate_covariance
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance


class EmpiricalSemivariogram:
    """
    Class calculates Experimental Semivariogram and Experimental Covariogram of a given dataset.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value) or (Point(), value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    weights : numpy array or None, optional, default=None
              weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted.

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

    is_semivariance : bool, optional, default=True
                      should semivariance be calculated?

    is_covariance : bool, optional, default=True
                    should covariance be calculated?

    is_variance : bool, optional, default=True
                  should variance be calculated?

    Attributes
    ----------
    input_array : numpy array
                  The array with coordinates and observed values.

    experimental_semivariance : numpy array or None, optional, default=None
                                The array of semivariance per lag in the form:
                                (lag, semivariance, number of points within lag).

    experimental_covariance : numpy array or None, optional, default=None
                              The array of covariance per lag in the form:
                              (lag, covariance, number of points within lag).

    variance : float or None, optional, default=None
               The variance of a dataset, if data is second-order stationary then we are able to retrieve a semivariance
               as a difference between the variance and the experimental covariance:

                    (Eq. 1)

                        g(h) = c(0) - c(h)

                        where:

                        g(h): semivariance at a given lag h,
                        c(0): variance of a dataset,
                        c(h): covariance of a dataset.

                Important! Have in mind that it works only if process is second-order stationary (variance is the same
                for each distance bin) and if the semivariogram has the upper bound.

    step : float
        Derived from the step_size parameter.

    mx_rng : float
        Derived from the  max_range parameter.

    weights : numpy array or None
        Derived from the weights paramtere.

    direct: float
        Derived from the direction parameter.

    tol : float
        Derived from the tolerance parameter.

    Methods
    -------
    __str__()
        prints basic info about the class parameters.

    __repr__()
        reproduces class initialization with an input data.

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
    ...])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = EmpiricalSemivariogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+--------------------+
    | lag |    semivariance    |      covariance     |    var_cov_diff    |
    +-----+--------------------+---------------------+--------------------+
    | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
    | 2.0 | 5.2272727272727275 | -0.7954545454545454 | 5.0439752555137165 |
    | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
    +-----+--------------------+---------------------+--------------------+
    """

    def __init__(self, input_array, step_size: float, max_range: float, weights=None, direction=0, tolerance=1,
                 is_semivariance=True, is_covariance=True, is_variance=True):

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        self.input_array = input_array
        self.experimental_semivariance = None
        self.experimental_covariance = None
        self.variance = None

        self.step = step_size
        self.mx_rng = max_range
        self.weights = weights
        self.direct = direction
        self.tol = tolerance

        if is_semivariance:
            self._calculate_semivariance()
        if is_covariance:
            self._calculate_covariance(is_variance)

        self.__c_sem = is_semivariance
        self.__c_cov = is_covariance
        self.__c_var = is_variance

    def _calculate_semivariance(self):
        """
        Method calculates semivariance.

        See: calculate_semivariance function.
        """
        self.experimental_semivariance = calculate_semivariance(
            points=self.input_array.copy(),
            step_size=self.step,
            max_range=self.mx_rng,
            weights=self.weights,
            direction=self.direct,
            tolerance=self.tol
        )

    def _calculate_covariance(self, get_variance=False):
        """
        Method calculates covariance and variance.

        See : calculate_covariance function.
        """
        self.experimental_covariance, self.variance = calculate_covariance(
            points=self.input_array.copy(),
            step_size=self.step,
            max_range=self.mx_rng,
            direction=self.direct,
            tolerance=self.tol,
            get_c0=get_variance
        )

    def __str_empty(self):
        if not self.__c_var:
            return "Empty object"
        else:
            return f"Variance: {self.variance:.4f}"

    def __str_populate_both(self):
        rows = []
        if self.__c_var:
            for idx, row in enumerate(self.experimental_semivariance):
                lag = row[0]
                smv = row[1]
                cov = self.experimental_covariance[idx][1]
                var_cov_diff = self.variance - cov
                rows.append([lag, smv, cov, var_cov_diff])
        else:
            for idx, row in enumerate(self.experimental_semivariance):
                lag = row[0]
                smv = row[1]
                cov = self.experimental_covariance[idx][1]
                rows.append([lag, smv, cov, nan])
        return rows

    def __str_populate_single(self):
        rows = []
        if self.__c_cov:
            if self.__c_var:
                for row in self.experimental_covariance:
                    lag = row[0]
                    cov = row[1]
                    var_cov_diff = self.variance - cov
                    rows.append([lag, nan, cov, var_cov_diff])
            else:
                for row in self.experimental_covariance:
                    lag = row[0]
                    cov = row[1]
                    rows.append([lag, nan, cov, nan])
        else:
            for row in self.experimental_semivariance:
                lag = row[0]
                sem = row[1]
                rows.append([lag, sem, nan, nan])
        return rows

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

    def __repr__(self):
        cname = 'EmpiricalSemivariogram'
        input_params = f'input_array={self.input_array.tolist()}, step_size={self.step}, max_range={self.mx_rng}, ' \
                       f'weights={self.weights}, direction={self.direct}, tolerance={self.tol}, ' \
                       f'is_semivariance={self.__c_sem}, is_covariance={self.__c_cov}, is_variance={self.__c_var}'
        repr_val = cname + '(' + input_params + ')'
        return repr_val


def build_experimental_variogram(input_array,
                                 step_size: float,
                                 max_range: float,
                                 weights=None,
                                 direction=0,
                                 tolerance=1) -> EmpiricalSemivariogram:
    """
    Function prepares:
        - experimental semivariogram,
        - experimental covariogram,
        - variance.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value) or (Point(), value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    weights : numpy array or None, optional, default=None
              weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted.

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

    Returns
    -------
    semivariogram_stats : EmpiricalSemivariogram
        The class with empirical semivariogram, empirical covariogram and variance

    See Also
    --------
    calculate_covariance : function to calculate experimental covariance and variance of a given set of points.
    calculate_semivariance : function to calculate experimental semivariance from a given set of points.
    EmpiricalSemivariogram : class that calculates and stores experimental semivariance, covariance and variance.

    Notes
    -----
    Function is an alias for EmpiricalSemivariogram class and it forces calculations of all spatial statistics from a
        given dataset.

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
    ...])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = build_experimental_variogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+--------------------+
    | lag |    semivariance    |      covariance     |    var_cov_diff    |
    +-----+--------------------+---------------------+--------------------+
    | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
    | 2.0 | 5.2272727272727275 | -0.7954545454545454 | 5.0439752555137165 |
    | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
    +-----+--------------------+---------------------+--------------------+

    """
    semivariogram_stats = EmpiricalSemivariogram(
        input_array=input_array,
        step_size=step_size,
        max_range=max_range,
        weights=weights,
        direction=direction,
        tolerance=tolerance,
        is_semivariance=True,
        is_covariance=True,
        is_variance=True
    )
    return semivariogram_stats
