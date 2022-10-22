"""
Additional exceptions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""


class IndexColNotUniqueError(Exception):
    """
    Raised when given index column is not unique.

    Parameters
    ----------
    dlength : int
              Dataset length (number of all records).

    number_of_unique_indices : int

    Attributes
    ----------
    msg : str
    """
    def __init__(self, dlength: int, number_of_unique_indices: int):
        ratio = round(100 * (number_of_unique_indices / dlength), 2)
        self.msg = f'Your dataset has {dlength} records but {number_of_unique_indices} unique indices are ' \
                   f'available which is {ratio}% of a dataset.'

    def __str__(self):
        return self.msg


class WrongGeometryTypeError(Exception):
    """
    Exception raised when given geometry is different than Polygon or MultiPolygon.

    Parameters
    ----------
    gtype : Any
            Wrong geometry type.

    Attributes
    ----------
    msg : str
    """

    def __init__(self, gtype):
        self.msg = f'Wrong type of geometry detected. You may use Polygon or MultiPolygon types, but' \
                   f' {gtype} was passed.'

    def __str__(self):
        return self.msg


class SetDifferenceWarning(Warning):
    """
    Warning used when two sets of values differs (can check if Blocks ids and PointSupport ids are the same).
    """

    def __init__(self, set_1_to_2_diff, set_2_to_1_diff, set_1_name, set_2_name):
        self.msg = f'Given sets of ids are not equal.' \
                   f' * Differences between {set_1_name} and {set_2_name} are {list(set_1_to_2_diff)}.' \
                   f' * Differences between {set_2_name} and {set_1_name} are {list(set_2_to_1_diff)}.'

    def __str__(self):
        return repr(self.msg)
