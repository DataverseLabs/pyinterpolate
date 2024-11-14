from typing import Optional
from pydantic import BaseModel, ConfigDict, FiniteFloat
from pyinterpolate.core.data_models.custom_types import ndarray_pydantic


class ExperimentalVariogramModel(BaseModel):
    """
    Class represents Experimental Variogram baseline model
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
