import numpy as np
from pydantic import BaseModel, validator


class PointsDataModel(BaseModel):
    """
    Model of points data
    """
    dataset: np.ndarray

    @validator("dataset")
    def has_minimal_number_of_columns(cls, v):
        if v.shape[1] < 3:
            raise AttributeError(
                'Provided array should have minimum three columns [x, y, value]'
            )
        return v
