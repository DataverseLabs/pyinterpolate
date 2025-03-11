from typing import Optional
from pydantic import BaseModel, ConfigDict, FiniteFloat
from pyinterpolate.core.validators.custom_types import ndarray_pydantic


class ExperimentalVariogramModel(BaseModel):
    """
    Class represents Experimental Variogram model

    Attributes
    ----------
    lags : numpy array, optional
        Lags in the experimental variogram model.

    points_per_lag : numpy array, optional
        Number of point pairs in each lag.

    semivariances : numpy array, optional
        Experimental semivariances.

    covariances : numpy array, optional
        Experimental covariances.

    variance : float, optional
        Experimental variance.

    direction : float, optional
        Direction of the experimental variogram.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at
        a single line with the beginning in the origin
        of the coordinate system and the direction given by
        y-axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is
        selected as an elliptical area with major axis
        pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional
          semivariogram.

    max_range : float, optional
        Maximum range of the experimental variogram.

    step_size : float, optional
        Step size in the experimental variogram
        (for evenly-spaced lags).

    custom_weights : numpy array, optional
        Custom weights for the experimental semivariances.
    """
    lags: Optional[ndarray_pydantic] = None
    points_per_lag: Optional[ndarray_pydantic] = None
    semivariances: Optional[ndarray_pydantic] = None
    covariances: Optional[ndarray_pydantic] = None
    variance: Optional[FiniteFloat] = None
    direction: Optional[FiniteFloat] = None
    tolerance: Optional[FiniteFloat] = None
    max_range: Optional[FiniteFloat] = None
    step_size: Optional[FiniteFloat] = None
    custom_weights: Optional[ndarray_pydantic] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
