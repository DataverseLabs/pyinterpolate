class AttributeSetToFalseWarning(Warning):
    """
    Warning invoked when ``ExperimentalVariogram`` class attributes are set
    to ``False`` (``is_semivariance``, ``is_covariance``) but user wants
    to plot one of the indices controlled by those attributes
    (semivariance, covariance) with the ``plot()`` method.
    """
    def __init__(self, validated):
        wrong_params = list(validated.keys())
        msg = ''
        for _param in wrong_params:
            attr_msg = (f'Warning! Attribute {_param} is set to False '
                        f'but you try to plot this object! Plot has been'
                        f' cancelled.')
            msg = msg + attr_msg
        self.message = msg

    def __str__(self):
        return repr(self.message)
