from typing import Union
import numpy as np
from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def build_theoretical_variogram(
    experimental_variogram: Union[ExperimentalVariogram, np.ndarray],
    models_group: Union[str, list] = 'safe',
    nugget=None,
    min_nugget=0,
    max_nugget=0.5,
    number_of_nuggets=16,
    rang=None,
    min_range=0.1,
    max_range=0.5,
    number_of_ranges=16,
    sill=None,
    min_sill=0.,
    max_sill=1,
    number_of_sills=16,
    direction=None,
    error_estimator='rmse',
    deviation_weighting='equal'
) -> TheoreticalVariogram:
    """
    Function creates Theoretical Variogram.

    Parameters
    ----------
    experimental_variogram : ExperimentalVariogram
        Experimental variogram model or array with lags and semivariances.

    models_group : str or list, default='safe'
            Models group to test:

            - 'all' - the same as list with all models,
            - 'safe' - ['linear', 'power', 'spherical']
            - as a list: multiple model types to test
            - as a single model type from:
                - 'circular',
                - 'cubic',
                - 'exponential',
                - 'gaussian',
                - 'linear',
                - 'power',
                - 'spherical'.

    nugget : float, optional
        Nugget (bias) of a variogram. If given then it is
        fixed to this value.

    min_nugget : float, default = 0
        The minimum nugget as the ratio of the parameter to
        the first lag variance.

    max_nugget : float, default = 0.5
        The maximum nugget as the ratio of the parameter to
        the first lag variance.

    number_of_nuggets : int, default = 16
        How many equally spaced nuggets tested between
        ``min_nugget`` and ``max_nugget``.

    rang : float, optional
        If given, then range is fixed to this value.

    min_range : float, default = 0.1
        The minimal fraction of a variogram range,
        ``0 < min_range <= max_range``.

    max_range : float, default = 0.5
        The maximum fraction of a variogram range,
        ``min_range <= max_range <= 1``. Parameter ``max_range`` greater
        than **0.5** raises warning.

    number_of_ranges : int, default = 16
        How many equally spaced ranges are tested between
        ``min_range`` and ``max_range``.

    sill : float, default = None
        If given, then sill is fixed to this value.

    min_sill : float, default = 0
        The minimal fraction of the variogram variance at lag 0 to
        find a sill, ``0 <= min_sill <= max_sill``.

    max_sill : float, default = 1
        The maximum fraction of the variogram variance at lag 0 to find
        a sill. It *should be* lower or equal to 1.
        It is possible to set it above 1, but then warning is printed.

    number_of_sills : int, default = 16
        How many equally spaced sill values are tested between
        ``min_sill`` and ``max_sill``.

    direction : float, in range [0, 360], default=None
        The direction of a semivariogram. If ``None`` given then
        semivariogram is isotropic. This parameter is required if
        passed experimental variogram is stored in a numpy array.

    error_estimator : str, default = 'rmse'
        A model error estimation method. Available options are:

        - 'rmse': Root Mean Squared Error,
        - 'mae': Mean Absolute Error,
        - 'bias': Forecast Bias,
        - 'smape': Symmetric Mean Absolute Percentage Error.

    deviation_weighting : str, default = "equal"
        The name of the method used to weight error at a given lags. Works
        only with RMSE. Available methods:

        - equal: no weighting,
        - closest: lags at a close range have bigger weights,
        - distant: lags that are further away have bigger weights,
        - dense: error is weighted by the number of point pairs within lag.

    Returns
    -------
    theo_var : TheoreticalVariogram
        Fitted theoretical semivariogram.
    """
    theo_var = TheoreticalVariogram()
    theo_var.autofit(
        experimental_variogram=experimental_variogram,
        models_group=models_group,
        nugget=nugget,
        min_nugget=min_nugget,
        max_nugget=max_nugget,
        number_of_nuggets=number_of_nuggets,
        rang=rang,
        min_range=min_range,
        max_range=max_range,
        number_of_ranges=number_of_ranges,
        sill=sill,
        min_sill=min_sill,
        max_sill=max_sill,
        number_of_sills=number_of_sills,
        direction=direction,
        error_estimator=error_estimator,
        deviation_weighting=deviation_weighting,
        return_params=False
    )
    return theo_var
