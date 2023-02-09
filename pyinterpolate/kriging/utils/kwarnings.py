"""
Kriging warnings.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
class ZerosMatrixWarning(Warning):
    """
    Warning invoked if any matrix in kriging system is populated with zeros.
    """

    def __init__(self):
        self.message = 'Matrix in your Kriging system is populated only be zeros. It is probably data error.' \
                       ' Prediction will return 0 as a predicted value and np.nan as an error.'

    def __str__(self):
        return repr(self.message)


class LeastSquaresApproximationWarning(Warning):
    """
    Warning used when algorithm searches for approximate solution instead of exact solution.
    """

    def __init__(self):
        self.message = 'Kriging system solution is based on the approximate solution, output may be incorrect!'

    def __str__(self):
        return repr(self.message)


class ExperimentalFeatureWarning(Warning):
    """
    Class describes experimental feature warnings.
    """

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
