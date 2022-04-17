import numpy as np


def check_weights(weights: np.ndarray):
    if np.any(weights[weights < 0]):
        pass


class NegativeWeightsWarning(Warning):
    """Warning invoked by the scenario when predicted value is equal to 0 and observation is equal to 0. It leads to
        the 0/0 division and, in return, to NaN value at a specific position. Finally, user gets NaN as the output.

    Parameters
    ----------
    message : str

    Attributes
    ----------
    message : str
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)