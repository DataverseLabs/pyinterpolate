from typing_extensions import Annotated

import numpy as np
from pydantic import BeforeValidator, PlainSerializer


def nd_array_custom_before_validator(x):
    # custom before validation logic
    return x


def nd_array_custom_serializer(x):
    # custom serialization logic
    return str(x)


ndarray_pydantic = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]
