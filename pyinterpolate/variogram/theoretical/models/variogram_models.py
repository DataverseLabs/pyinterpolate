import numpy as np


def circular_model(lags: np.array, nugget: float, sill: float, range: float) -> np.array:
    """Function calculates circular model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    range : float

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    gamma = nugget + (2/np.pi)*sill*[a * np.sqrt(1 - a ** 2) + np.arcsin(a)], 0 < lag <= range
        gamma = 0, lag == 0
        (Source: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-kriging-works.htm#GUID-94A34A70-DBCF-4B23-A198-BB50FB955DC0))

        where:

        a = lag / range


        NOTE: There exists an equivalent model form for the circular model:
        gamma = nugget + sill*[1 - (2/np.pi * np.arccos(a)) + (2/np.pi * a) * np.sqrt(1 - a ** 2 )], 0 < lag <= range
        gamma = 0, lag == 0
        (Source: 'McBratney, A. B., and R. Webster. "Choosing Functions for Semi-variograms of Soil Properties and Fitting Them to Sampling Estimates." Journal of Soil Science 37: 617–639. 1986.')

        (Convert between the two forms using: arcsin(x) + arccos(x) = np.pi/2. Then, adjust pre-factors outside of bracket term).


    """