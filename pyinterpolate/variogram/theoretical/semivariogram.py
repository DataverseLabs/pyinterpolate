from pyinterpolate.variogram.empirical import EmpiricalVariogram


class TheoreticalVariogram:
    """Theoretical model of spatial data dissimilarity.

    Parameters
    ----------
    verbose : bool, default=False
              Prints messages related to the model preparation.

    Attributes
    ----------
    observations : numpy array
                   Coordinates and their values: (pt x, pt y, value) or (Point(), value).

    verbose : bool, default=False
              Prints messages related to the model preparation.

    empirical_variogram : EmpiricalVariogram or None, default=None
                          Empirical Variogram class and its attributes.

    theoretical_model : str or None, default=None
                        The name of a theoretical model. Only finite set of models are available.
                        See variogram_models attribute.

    variogram_models : dict
                       Dict with keys representing theoretical variogram models and values that
                       are pointing into a modeling methods. Available models:
                           'circular',
                           'cubic',
                           'exponential',
                           'gaussian',
                           'linear',
                           'power',
                           'spherical'.

    name : str or None, default=None
           Name of the chosen model. Available names are the same as keys in variogram_models attribute.

    nugget : float, default=0
             Nugget parameter (bias at a zero distance).

    sill : float, default=0
           Value at which dissimilarity is close to its maximum if model is bounded. Otherwise, it is usually close
           to observations variance.

    range : float, default=0
            Range is a distance at which spatial correlation exists and often it is a distance when variogram reaches
            sill. It shouldn't be set at a distance larger than a half of a study extent.

    rmse : float, default=0
           Root mean squared error of the difference between the empirical observations and the modeled curve.

    bias : float, default=0
           Forecast Bias of the estimation. Large positive value means that the estimated model usually overestimates
           values and large negative value means that model underestimates predictions.

    smape : float, default=0
            Symmetric Mean Absolute Percentage Error of the prediction - values from 0 to 100%.

    akaike : float, default=0
             Akaike information criterion (AIC) of a given model. Quality of a model.

    """

    def __init__(self,
                 empirical_model=None,
                 verbose=False):

        self.verbose = verbose

        # Model
        self.observations = None
        self.empirical_variogram = empirical_model
        self.variogram_models = {
            'circular': self.circular_model,
            'cubic': self.cubic_model,
            'exponential': self.exponential_model,
            'gaussian': self.gaussian_model,
            'linear': self.linear_model,
            'power': self.power_model,
            'spherical': self.spherical_model
        }

        # Model parameters
        self.model = None
        self.name = None
        self.nugget = 0.
        self.range = 0.
        self.sill = 0.

        # Dynamic parameters
        self.rmse = 0.
        self.bias = 0.
        self.smape = 0.
        self.akaike = 0.

    def set_empirical_model(self,
                            empirical_model=None,
                            observations=None,
                            step_size=None,
                            max_range=None,
                            weights=None,
                            direction=0,
                            tolerance=1,
                            auto_search=False):
        """Method sets empirical model for further calculations.

        Parameters
        ----------
        empirical_model : EmpiricalVariogram or None, default=None
                          Prepared EmpiricalVariogram model or None. If None then the observations parameter must
                          be given along with other semivariogram parameters.

        observations : numpy array or None, default=None
                       Coordinates and their values: (pt x, pt y, value) or (Point(), value). If None then
                       the EmpiricalVariogram parameter must be given.

        step_size :
        max_range
        weights
        direction
        tolerance
        autofit

        Returns
        -------

        """

    def __str__(self):
        pass

    def __repr__(self):
        pass
