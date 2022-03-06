import numpy as np


def circular_model(lags: np.array, nugget: float, sill: float, srange: float) -> np.array:
    """Function calculates circular model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    srange : float
             Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c[1 - (\frac{2}{\pi} * arccos(a)) + (\frac{2}{\pi} * a) * \sqrt{1 - a^{2}}]$, $0 < a <= h$;
    (2) $\gamma = c0 + c$, $a > h$;
    (3) $\gamma = 0$, $a = 0$.

    where:

    - $\gamma$ - semivariance,
    - $c0$ - nugget,
    - $c$ - sill,
    - $a$ - lag,
    - $h$ - range.

    Bibliography
    ------------
    [1] McBratney, A. B., Webster R. Choosing Functions for Semivariograms of Soil Properties and Fitting Them to
    Sampling Estimates. Journal of Soil Science 37: 617–639. 1986.


    """

    pic = 2 / np.pi

    gamma = np.where(
        (lags <= srange),
        (nugget + sill*(1 - (pic * np.arccos(lags)) + (pic * lags) * np.sqrt(1 - lags**2))),
        (nugget + sill)
    )

    if lags[0] == 0:
        gamma[0] = 0
    return gamma
