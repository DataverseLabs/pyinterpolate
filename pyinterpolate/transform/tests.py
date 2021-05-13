def does_variogram_exist(theoretical_model):
    """
    Function checks if variogram is calculated.

    INPUT:

    :param theoretical_model: (TheoreticalSemivariogram)

    OUTPUT:

    :return: (bool) True if Theoretical Semivariogram has calculated params or False otherwise
    """
    # Test
    if (theoretical_model.nugget is None) or (theoretical_model.sill is None) or (theoretical_model.range is None):
        raise ValueError('Nugget, sill or range of TheoreticalSemivariogram is not set. Please update '
                         'TheoreticalSemivariogram model before you pass it into Kriging model.')
