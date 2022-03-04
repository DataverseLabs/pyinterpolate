class ErrorTypeSelectionError(Exception):
    """Error invoked if user doesn't select any error type for the theoretical variogram modeling.

    Attributes
    ----------
    message : str
    """
    def __init__(self):
        self.message = "You didn't selected any error type from available rmse, bias, akaike and smape. Set one of" \
                       " those to True."

    def __str__(self):
        return self.message


def check_selected_errors(val: int):
    if val == 0:
        raise ErrorTypeSelectionError
    else:
        pass