import matplotlib.pyplot as plt


class Deconvolution:
    """
    Class performs deconvolution of semivariogram of areal data. Whole procedure is based on the iterative process
    described in: [1].

    Steps to regularize semivariogram:
    - initialize your object (no parameters),
    - use fit() method to build initial point support model,
    - use transform() method to perform semivariogram regularization,
    - save semivariogram model with export_model() method.

    Attributes
    ----------
    iter : int
           A control parameter. Number of iterations.

    iters_max : int
                A control parameter. Maximum number of iterations.

    deviation_ratio : float
                      A control parameter. Ratio of the initial regularization error and the last iteration
                      regularization error. Regularization error is the Mean Absolute Error between the regularized
                      areal semivariogram and the point-support theoretical semivariance. Smaller ratio > better model.

    min_deviation_ratio : float
                          A control parameter. The minimal deviation ratio when algorithm stops.





    Methods
    -------
    fit()
        Fits areal data and the point support data into a model, initializes the experimental semivariogram,
        the theoretical semivariogram model, regularized point support model, and deviation.

    transform()
        Performs semivariogram regularization.

    export_model()
        Exports regularized (or fitted) model.

    import_model()
        Imports regularized (or fitted) model.

    plot()
        Plots semivariances before and after regularization.


    References
    ----------
    [1] Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical
    Units, Mathematical Geology 40(1), 101-128, 2008

    Examples
    --------


    """

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        pass

    def import_model(self):
        pass

    def export_model(self):
        pass

    def plot(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass