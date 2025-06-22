def singular_matrix_error():
    """
    Raises RuntimeError with message about singular matrix in Kriging system.

    Raises
    -------
    RuntimeError
        Singular matrix in the Kriging system.
    """
    msg = ("Singular matrix in Kriging system detected, "
           "check if you have duplicated coordinates "
           "in the ``known_locations`` variable. If your "
           "data doesn't have duplicates then set "
           "``allow_lsa`` parameter to ``True``.")
    raise RuntimeError(msg)
