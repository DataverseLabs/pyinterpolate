from typing import Union, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from pyinterpolate.kriging.point_kriging import kriging
from pyinterpolate.variogram.empirical.experimental_variogram import (
    ExperimentalVariogram)
from pyinterpolate.variogram.theoretical.semivariogram import (
    TheoreticalVariogram)


class MultivariateRegression:
    """
    Class performs multivariate regression.

    Attributes
    ----------
    features : numpy array
        Features matrix.

    y : numpy array
        Predicted value.

    coefficients : numpy array
        Array of coefficients.

    intercept : float
        Constant bias.

    Methods
    -------
    fit()
        Fits observations into linear regression model.

    predict()
        Predicts values based on the coefficients from training (``fit()``)
        step.
    """

    def __init__(self):
        self.features = None
        self.y = None
        self.coefficients = None
        self.intercept = None

    def fit(self, dataset: np.ndarray):
        """
        Method fits dataset into the linear regression model.

        Parameters
        ----------
        dataset : numpy array
            Numpy array with more than 2 columns. The last column represents
            response - observations.
        """
        self.features = dataset[:, :-1].copy()
        self.y = dataset[:, -1].copy()
        _features_ones = np.c_[
            self.features, np.ones(self.features.shape[0])
        ]
        params = self._get_coefficients(_features_ones)
        self.coefficients = params[:-1]
        self.intercept = params[-1]

    def predict(self, features: np.ndarray):
        """
        Predicts response value based on given variables.

        Parameters
        ----------
        features : numpy array
            The variables used for prediction.

        Returns
        -------
        : numpy array
            Predicted values.
        """
        # TODO: check if features has the same number of columns
        #  as self.features
        return np.matmul(self.coefficients, features.T) + self.intercept

    def _get_coefficients(self, features_with_ones):
        return np.linalg.lstsq(features_with_ones, self.y, rcond=None)[0]


class UniversalKriging:
    """
    Performs universal kriging with semivariogram autofit.

    Parameters
    ----------
    observations : numpy array
        Known points and their values.

    fitted_regression_model : optional
        Any kind of regression model with `.predict()` method that could be
        used for trend modeling.

    Attributes
    ----------
    observations : numpy array
        See ``observations`` parameter.

    trend_model : MultivariateRegression or another regression model
        The model responsible for trend prediction. It could be a model from
        the external package, but it must have `.predict()` method which takes
        as an input modeling features (coordinates).

    trend_values : numpy array
        Observation values predicted by regression model.

    bias_values : numpy array
        The difference between measured observations and ``trend_values``.

    bias_experimental_model : ExperimentalVariogram
        The experimental semivariogram of bias.

    bias_model : TheoreticalVariogram
        Modeled semivariogram of the data bias.

    Methods
    -------
    fit_trend()
        Fits observations into linear regression model.

    detrend()
        Removes trend from observations (calculates bias).

    fit_bias()
        Fits bias into semivariogram model.

    predict()
        Predicts values at unknown locations.

    plot_experimental_bias_model()
        Plots experimental semivariogram of bias.

    plot_theoretical_bias_model()
        Plots theoretical semivariogram of bias.

    plot_trend_surfaces()
        Visual comparison of observations, trend, and bias.
    """

    def __init__(self,
                 observations: np.ndarray,
                 fitted_regression_model: Any = None):
        # Core
        self.observations = observations
        # Trend
        self.trend_model = None
        self.trend_values = None
        if fitted_regression_model is not None:
            self.trend_model = fitted_regression_model
            self.trend_values = self.trend_model.predict(
                self.observations[:, :-1]
            )

        # Bias
        self.bias_values = None
        self.bias_experimental_model: ExperimentalVariogram = None
        self.bias_model: TheoreticalVariogram = None

    def fit_trend(self):
        """
        Function fits data into MultivariateRegression model.
        """
        trend_model = MultivariateRegression()
        trend_model.fit(self.observations)
        self.trend_model = trend_model
        self.trend_values = self.trend_model.predict(
            features=self.observations[:, :-1]
        )

    def detrend(self, trend_values: ArrayLike = None):
        """
        Function removes trend from observations.

        Parameters
        ----------
        trend_values : optional, numpy array
            Optional array with trend values. It may be an output from the
            external regression model.

        """
        if trend_values is not None:
            self.bias_values = self.observations[:, -1] - trend_values
        else:
            self.bias_values = self.observations[:, -1] - self.trend_values

    def fit_bias(self,
                 step_size: float,
                 max_range: float,
                 use_all_models=False):
        """
        Function fits bias into variogram models.

        Parameters
        ----------
        step_size : float
            Bins width.

        max_range : float
            Maximum range of analysis.

        use_all_models : bool, default = False
            Use all available semivariogram models (``True``), or only
            the selection of models - linear, spherical, and power model.
        """
        # Get bias
        if self.bias_values is None:
            self.detrend()

        # Create experimental semivariogram
        bias_arr = np.c_[
            self.observations[:, :-1], self.bias_values
        ]
        self.bias_experimental_model = ExperimentalVariogram(
            input_array=bias_arr,
            step_size=step_size,
            max_range=max_range,
            is_semivariance=True,
            is_covariance=False
        )

        # Fit experimental into theoretical
        if use_all_models:
            used_models = 'all'
        else:
            used_models = 'safe'

        self.bias_model = TheoreticalVariogram()
        self.bias_model.autofit(
            experimental_variogram=self.bias_experimental_model,
            model_name=used_models
        )

    def predict(self,
                points: ArrayLike,
                how: str = 'ok',
                neighbors_range: Union[float, None] = None,
                no_neighbors: int = 4,
                use_all_neighbors_in_range=False,
                sk_mean: Union[float, None] = None,
                allow_approx_solutions=False,
                number_of_workers: int = 1,
                show_progress_bar: bool = True):
        """

        Parameters
        ----------
        points : numpy array
            Coordinates with missing values (to estimate results).

        how : str, default='ok'
            Select what kind of kriging you want to perform:
              * 'ok': ordinary kriging,
              * 'sk': simple kriging - if it is set then ``sk_mean`` parameter
                must be provided.

        neighbors_range : float, default=None
            The maximum distance where we search for neighbors. If ``None``
            is given then range is selected from
            the ``theoretical_model`` ``rang`` attribute.

        no_neighbors : int, default = 4
            The number of the **n-closest neighbors** used for interpolation.

        use_all_neighbors_in_range : bool, default = False
            ``True``: if the real number of neighbors within the
            ``neighbors_range`` is greater than the
            ``number_of_neighbors`` parameter then take all of them anyway.

        sk_mean : float, default=None
            The mean value of a process over a study area. Should be know
            before processing. That's why Simple
            Kriging has a limited number of applications. You must have
            multiple samples and well-known area to
            know this parameter.

        allow_approx_solutions : bool, default=False
            Allows the approximation of kriging weights based on the OLS
            algorithm. We don't recommend set it to ``True``
            if you don't know what are you doing. This parameter can be useful
            when you have clusters in your dataset,
            that can lead to singular or near-singular matrix creation.

        number_of_workers : int, default=1
            How many processing units can be used for predictions. Increase it
            only for a very large number of
            interpolated points (~10k+).

        show_progress_bar : bool, default=True
            Show progress bar of predictions.

        Returns
        -------
        : numpy array
            Predictions ``[predicted value, longitude (x), latitude (y)]``
        """
        # Predict trend surface from regression model
        trends = self.trend_model.predict(
            points
        )

        # Predict bias
        biases = kriging(
            observations=np.c_[self.observations[:, :-1], self.bias_values],
            theoretical_model=self.bias_model,
            points=points,
            how=how,
            neighbors_range=neighbors_range,
            no_neighbors=no_neighbors,
            use_all_neighbors_in_range=use_all_neighbors_in_range,
            sk_mean=sk_mean,
            allow_approx_solutions=allow_approx_solutions,
            number_of_workers=number_of_workers,
            show_progress_bar=show_progress_bar
        )
        biases = biases[:, 0]

        # parse predictions
        predicted_values = trends + biases
        predicted = np.c_[
            predicted_values, points[:, 0], points[:, 1]
        ]
        return predicted

    def plot_experimental_bias_model(self):
        """
        Plots experimental variogram of bias.
        """
        self.bias_experimental_model.plot(
            plot_semivariance=True,
            plot_covariance=False,
            plot_variance=False
        )

    def plot_theoretical_bias_model(self):
        """
        Plots theoretical semivariogram of bias.
        """
        self.bias_model.plot()

    def plot_trend_surfaces(self):
        """
        Plots initial observations, trend surface, and bias surface.
        """
        fig, ax = plt.subplots(3)

        # Base surface
        base_x = self.observations[:, 0]
        base_y = self.observations[:, 1]
        base_z = self.observations[:, 2]
        base_min = np.min(base_z)
        base_max = np.max(base_z)
        base_std = np.std(base_z) * 2

        base_plot = ax[0].scatter(
            x=base_x,
            y=base_y,
            c=base_z,
            cmap='gist_earth',
            vmin=base_min,
            vmax=base_max
        )

        fig.colorbar(mappable=base_plot, ax=ax[0])

        # trend surface
        trend_plot = ax[1].scatter(
            x=base_x,
            y=base_y,
            c=self.trend_values,
            cmap='gist_earth',
            vmin=base_min,
            vmax=base_max
        )

        fig.colorbar(mappable=trend_plot, ax=ax[1])

        # error
        bias_plot = ax[2].scatter(
            x=base_x,
            y=base_y,
            c=base_z - self.trend_values,
            cmap='cool',
            vmin=-base_std,
            vmax=base_std
        )

        fig.colorbar(mappable=bias_plot, ax=ax[2])

        plt.show()
