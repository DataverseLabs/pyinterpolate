from pyinterpolate.variogram.empirical.covariance import calculate_covariance
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance


class EmpiricalSemivariogram:

    def __init__(self, input_array, step_size: float, max_range: float, weights=None, direction=0, tolerance=0,
                 calculate_semivariance=True, calculate_covariance=False, get_variance_c0=False):

        self.input_array = input_array
        self.experimental_semivariance = None
        self.experimental_covariance = None
        self.variance = None

        self.step = step_size
        self.mx_rng = max_range
        self.weights = weights
        self.direct = direction
        self.tol = tolerance

        if calculate_semivariance:
            self._calculate_semivariance()
        if calculate_covariance:
            self._calculate_covariance(get_variance_c0)


        self.is_semivariance = calculate_semivariance
        self.is_covariance = calculate_covariance
        self.is_variance = get_variance_c0

    def _calculate_semivariance(self):
        self.experimental_semivariance = calculate_semivariance(
            points=self.input_array.copy(),
            step_size=self.step,
            max_range=self.mx_rng,
            weights=self.weights,
            direction=self.direct,
            tolerance=self.tol
        )

    def _calculate_covariance(self, get_variance=False):
        self.experimental_covariance, self.variance = calculate_covariance(
            points=self.input_array.copy(),
            step_size=self.step,
            max_range=self.mx_rng,
            direction=self.direct,
            tolerance=self.tol,
            get_c0=get_variance
        )
