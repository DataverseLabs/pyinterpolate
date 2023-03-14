from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Tuple, Optional

import numpy as np

from pyinterpolate import TheoreticalVariogram


class KrigingType(Enum):
    ORDINARY = 0
    SIMPLE = 1


@dataclass
class KrigingObject:
    """
    Representation of Kriging object that can be stored and reused.
    """
    type: KrigingType
    theoretical_model: TheoreticalVariogram
    known_locations: np.ndarray
    unknown_location: Union[List, Tuple, np.ndarray]
    no_neighbors: int
    neighbors_range: Optional[float]
    process_mean: Optional[float]
    use_all_neighbors_in_range: Optional[bool]
    allow_approximate_solutions: Optional[bool]