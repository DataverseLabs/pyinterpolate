from typing import Optional
from pydantic import BaseModel, ConfigDict

from pyinterpolate.core.validators.custom_types import ndarray_pydantic
from pyinterpolate.core.data_models.experimental_variogram import ExperimentalVariogramModel


class SemivariogramErrorsModel(BaseModel):
    """
    Error types used for automatic fitting of semivariogram models.
    """
    bias: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    smape: Optional[float] = None


class TheoreticalVariogramModel(BaseModel):
    """
    Class represents Theoretical Variogram baseline model
    """
    experimental_variogram: Optional[ExperimentalVariogramModel] = None
    nugget: float
    sill: float
    rang: float
    variogram_model_type: str
    direction: Optional[float] = None
    spatial_dependence: Optional[str] = None
    spatial_index: Optional[float] = None
    yhat: Optional[ndarray_pydantic] = None
    errors: Optional[SemivariogramErrorsModel] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
