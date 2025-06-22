import numpy as np
from matplotlib import pyplot as plt

from pyinterpolate.evaluate.metrics import root_mean_squared_error
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def calculate_deviation(theoretical: np.ndarray,
                        regularized: np.ndarray,
                        method='mrd') -> float:
    """
    Function calculates deviation between the initial block semivariogram
    model and the regularized point support model.

    Parameters
    ----------
    theoretical : numpy array
        Semivariances.

    regularized : numpy array
        Semivariances.

    method : str, default=`'mrd'`
        Deviation monitoring method, available options:
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
        Semivariances calculated from the block Theoretical Model,
        array of length N.

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
        Semivariances calculated from the blocks Theoretical Model,
        array of length N.

    Returns
    -------
    deviation : float
        ``|Theoretical - Regularized| /
          [0.5 * (|Regularized| + |Theoretical|)]``
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
    Regularization process deviation calculation and monitoring.
    Deviation is defined as absolute difference between the theoretical
    semivariances and the regularized semivariances divided by the
    regularized semivariances.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted model.

    regularized_variances : numpy array
        ``[lag, semivariance]``

    method : str, default=`'mrd'`
        Deviation monitoring method, available options:

        * `'mrd'` - mean relative difference
        * `'smrd'` - symmetric mean relative difference
        * `'rmse'` - root mean squared error

    Attributes
    ----------
    method : str
        See ``method`` parameter.

    initial_deviation : float
        See ``initial_deviation`` parameter.

    deviations : list
        The list of deviations. The first element is the initial deviation.

    optimal_deviation : float
        Minimal deviation. During the first iteration it is equal to the
        initial deviation. It is updated only when the current deviation
        is lower than the optimal deviation.

    Methods
    -------
    current_deviation_decrease(), property
        Current deviation decrease (current deviation minus the lowest
        deviation).

    current_ratio(), property
        Ratio of current deviation to initial deviation.

    calculate_deviation_decrease()
        Current deviation decrease (current deviation minus the lowest
        deviation).

    calculate_deviation_ratio()
        Ratio of current deviation to initial deviation.

    deviation_direction()
        Returns True if deviation is increasing,
        False if deviation is decreasing.

    normalize()
        Ratios of current deviations to initial deviations. (Element-wise).

    plot()
        Plots the deviations.

    Raises
    ------
    KeyError : User provides unsupported deviation method name.

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

    @property
    def current_deviation_decrease(self):
        """
        Current deviation decrease (current deviation minus the lowest
        deviation).

        Returns
        -------
        : float
        """
        return self._current_deviation_decrease

    @property
    def current_ratio(self):
        """
        Ratio of current deviation to initial deviation.

        Returns
        -------
        : float
        """
        return self._current_deviation_ratio

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

    def deviation_direction(self) -> bool:
        """
        Returns ``False`` if deviation is decreasing,
        ``True`` if deviation is increasing.

        Returns
        -------
        : bool
        """

        if self.calculate_deviation_decrease() < 0:
            return False
        return True

    def normalize(self):
        """
        Normalizes the deviations by dividing them by the initial deviation.

        Returns
        -------
        : numpy array
            Normalized array of deviations.
        """
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

        plt.title(f'Deviation change, '
                  f'baseline deviation: {self.initial_deviation}')
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.show()

    def set_current_as_optimal(self):
        """
        Sets current deviation as optimal deviation. It is used when the
        current deviation is lower than the optimal deviation.
        """
        self.optimal_deviation = self.deviations[-1]

    def update(self,
               theoretical_model: np.ndarray,
               regularized_variances: np.ndarray):
        """
        Updates the deviation list with the current deviation.

        Parameters
        ----------
        theoretical_model : numpy array
            The initial semivariances.

        regularized_variances : numpy array
            Regularized semivariances.
        """
        deviation = calculate_deviation(theoretical_model,
                                        regularized_variances)
        self.deviations.append(deviation)

    def _check_deviation_method(self, method: str):
        if method in self._allowed_methods:
            return method
        else:
            raise KeyError(f'Deviation estimation method '
                           f'{method} is not supported, please use '
                           f'"mrd" - mean relative difference, '
                           f'"smrd" - symmetric mean relative difference, '
                           f'"rmse" - root mean squared error instead.')
