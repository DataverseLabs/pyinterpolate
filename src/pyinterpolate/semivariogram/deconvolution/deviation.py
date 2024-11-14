import numpy as np
from matplotlib import pyplot as plt

from pyinterpolate.evaluate.metrics import root_mean_squared_error
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def calculate_deviation(theoretical: np.ndarray,
                        regularized: np.ndarray,
                        method='mrd') -> float:
    """
    Function calculates deviation between initial block variogram model and the regularized point support model.

    Parameters
    ----------
    theoretical : numpy array
        Semivariances.

    regularized : numpy array
        Semivariances

    method : str, default=`'mrd'`
        Deviation dir_neighbors_selection_method, available options:
          * `'mrd'` - mean relative difference
          * `'smrd'` - symmetric mean relative difference
          * `'rmse'` - root mean squared error

    Returns
    -------
    deviation : float
    """
    if method == 'smrd':
        deviation = symmetric_mean_relative_difference(regularized,
                                                       theoretical)
    elif method == 'rmse':
        deviation = root_mean_squared_error(
            regularized,
            theoretical
        )
    else:
        deviation = mean_relative_difference(
            regularized, theoretical
        )
    return deviation


def mean_relative_difference(y_exp: np.ndarray, y_init: np.ndarray):
    """
    Function calculates deviation between regularized and modeled values.

    Parameters
    ----------
    y_exp : numpy array
        Output from model regularization, array of length N.

    y_init : numpy array
        Semivariances calculated from baseline Theoretical Model, array of length N.

    Returns
    -------
    deviation : float
        ``|Theoretical - Regularized| / Regularized``
    """

    # Todo: track regularized semivariances to ensure that they are != 0

    # Ensure that both arrays are floats
    y_exp = y_exp.astype(float)
    y_init = y_init.astype(float)

    # Calc
    numerator = np.abs(y_init - y_exp)
    deviations = numerator / y_exp
    deviation = float(np.mean(deviations))
    return deviation


def symmetric_mean_relative_difference(y_exp: np.ndarray, y_init: np.ndarray):
    """
    Function calculates deviation between regularized and modeled values.

    Parameters
    ----------
    y_exp : numpy array
        Output from model regularization, array of length N.

    y_init : numpy array
        Semivariances calculated from baseline Theoretical Model, array of length N.

    Returns
    -------
    deviation : float
        ``|Theoretical - Regularized| / [0.5 * (|Regularized| + |Theoretical|)]``
    """
    # Ensure that both arrays are floats
    y_exp = y_exp.astype(float)
    y_init = y_init.astype(float)

    # Calc
    numerator = np.abs(y_init - y_exp)
    denominator = (np.abs(y_init) + np.abs(y_exp)) / 2
    deviations = numerator / denominator
    deviation = float(np.mean(deviations))
    return deviation


class Deviation:
    """
    Regularization process deviation calculation and monitoring. Deviation in its base form
    is defined as ``|Theoretical semivariances - Regularized semivariances| / Regularized semivariances``.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram

    regularized_variances : numpy array
                  [lag, semivariance]

    method : str, default=`'mrd'`
        Deviation dir_neighbors_selection_method, available options:
          * `'mrd'` - mean relative difference
          * `'smrd'` - symmetric mean relative difference
          * `'rmse'` - root mean squared error

    Attributes
    ----------
    deviations : list
        Deviations of each iteration. The first element is the initial deviation.

    initial_deviation : float
        The initial deviation (

    Methods
    -------

    Raises
    ------
    KeyError : User provides not supported deviation dir_neighbors_selection_method

    """

    def __init__(self,
                 theoretical_model: np.ndarray,
                 regularized_variances: np.ndarray,
                 method: str = 'mrd'):

        self._allowed_methods = {'mrd', 'smrd', 'rmse'}
        self.method = self._check_deviation_method(method)

        self.initial_deviation = calculate_deviation(theoretical_model,
                                                     regularized_variances)
        self.deviations = [self.initial_deviation]

        self.optimal_deviation = self.initial_deviation

        self._current_deviation_decrease = self.calculate_deviation_decrease()
        self._current_deviation_ratio = self.calculate_deviation_ratio()

    def calculate_deviation_decrease(self):
        """
        Deviation decrease (current deviation minus the lowest deviation).

        Returns
        -------
        : float
        """

        dev_decrease = self.deviations[-1] - self.optimal_deviation
        return dev_decrease

    def calculate_deviation_ratio(self):
        """
        Ratio of current deviation to initial deviation.

        Returns
        -------
        : float
        """
        ratio = self.deviations[-1] / self.initial_deviation
        return ratio

    def current_deviation_decrease(self):
        return self._current_deviation_decrease

    def current_ratio(self):
        return self._current_deviation_ratio

    def deviation_direction(self) -> bool:
        """
        Returns ``False`` if deviation is decreasing, ``True`` if deviation is increasing.

        Returns
        -------
        : bool
        """

        if self.calculate_deviation_decrease() < 0:
            return False
        return True

    def normalize(self):
        return np.array(self.deviations) / self.initial_deviation

    def plot(self, normalized=True):
        plt.figure(figsize=(12, 6))

        iters = np.arange(len(self.deviations))

        if normalized:
            deviations = self.normalize()
            ylabel = 'Normalized deviation'
        else:
            deviations = self.deviations
            ylabel = 'Deviation'

        plt.plot(
            iters, deviations
        )

        plt.title(f'Deviation change, baseline deviation: {self.initial_deviation}')
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.show()

    def set_current_as_optimal(self):
        self.optimal_deviation = self.deviations[-1]

    def update(self,
               theoretical_model: np.ndarray,
               regularized_variances: np.ndarray):

        deviation = calculate_deviation(theoretical_model, regularized_variances)
        self.deviations.append(deviation)

    def _check_deviation_method(self, method: str):
        if method in self._allowed_methods:
            return method
        else:
            raise KeyError(f'Deviation dir_neighbors_selection_method {method} is not supported, please use "mrd" - mean relative difference, '
                           f'"smrd" - symmetric mean relative difference, "rmse" - root mean squared error instead.')

