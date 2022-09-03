"""
Variogram models.

Authors
-------
1. Scott Gallacher | @scottgallacher-3
2. Szymon Moliński | @SimonMolinsky
"""
import numpy as np


def _get_zero_lag_value(lag0: float, nugget: float, gamma0: float) -> float:
    """Function checks if lag zero should be larger than zero (nugget is provided & lag on a distance 0 exists).

    Parameters
    ----------
    lag0 : float

    nugget : float

    gamma0 : float
             Semivariance of a lag 0 estimated by a model.

    Returns
    -------
    float
    """
    if lag0 == 0:
        if nugget != 0:
            return nugget
        else:
            return 0
    return gamma0


def circular_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates circular model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
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
    ar = lags / rang
    ns = nugget + sill
    gamma = np.ones(len(ar)) * ns
    poses = lags <= rang
    arp = ar[poses]
    ar2 = np.power(arp, 2)
    gamma[poses] = nugget + sill * (1 - (pic * np.arccos(arp)) + (pic * arp) * np.sqrt(1 - ar2))

    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)

    return gamma


def cubic_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates cubic model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
           Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c *
         (7 * (\frac{a}{h})^{2} - 8.75 * (\frac{a}{h})^{3} + 3.5 * (\frac{a}{h})^{5} - 0.75 * (\frac{a}{h})^{7})$,
         $0 < a <= h$;, $0 < a <= h$;
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
    [1] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.

    """

    # ar = lags / rang
    # a1 = 7 * ar * ar
    # a2 = -8.75 * ar ** 3
    # a3 = 3.5 * ar ** 5
    # a4 = -0.75 * ar ** 7

    ar = lags / rang
    ns = nugget + sill
    gamma = np.ones(len(ar)) * ns
    poses = lags <= rang
    arp = ar[poses]

    arp2 = np.power(arp, 2) * 7
    arp3 = np.power(arp, 3) * -8.75
    arp5 = np.power(arp, 5) * 3.5
    arp7 = np.power(arp, 7) * -0.75
    arp_sum = arp2 + arp3 + arp5 + arp7

    gamma[poses] = nugget + sill * arp_sum

    # gamma = np.where(
    #     (lags <= rang),
    #     nugget + sill * (a1 + a2 + a3 + a4),
    #     nugget + sill
    # )

    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)

    return gamma


def exponential_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates exponential model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
           Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c * (1 - \exp({-\frac{a}{h}}))$, $a > 0$,
    (2) $\gamma = 0$, $a = 0$.

    where:

    - $\gamma$ - semivariance,
    - $c0$ - nugget,
    - $c$ - sill,
    - $a$ - lag,
    - $h$ - range.

    Bibliography
    ------------
    [1] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.

    """
    ar = lags / rang
    gamma = nugget + sill * (1 - np.exp(-ar))
    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)
    return gamma


def gaussian_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates gaussian model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
           Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c * (1 - \exp(-\frac{a^{2}}{h^{2}}))$, $a > 0$
    (2) $\gamma = 0$, $a = 0$.

    where:

    - $\gamma$ - semivariance,
    - $c0$ - nugget,
    - $c$ - sill,
    - $a$ - lag,
    - $h$ - range.

    Bibliography
    ------------
    [1] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.

    """

    rang_sq = rang * rang
    ar = lags**2 / rang_sq
    gamma = nugget + sill * (1 - np.exp(-ar))
    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)
    return gamma


def linear_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates linear model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
           Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c * \frac{a}{h}$, $0 < a <= h$;
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
    [1] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.

    """
    ar = lags / rang

    gamma = np.where(
        lags <= rang,
        nugget + sill * ar,
        nugget + sill
    )
    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)
    return gamma


def power_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates power model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
           Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c * (\frac{a}{h})^{2}$, $0 < a <= h$;
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
    [1] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.

    """
    ar = lags / rang
    ar_sq = ar * ar

    gamma = np.where(
        lags <= rang,
        nugget + sill * ar_sq,
        nugget + sill
    )
    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)
    return gamma


def spherical_model(lags: np.array, nugget: float, sill: float, rang: float) -> np.array:
    """Function calculates spherical model of semivariogram.

    Parameters
    ----------
    lags : numpy array

    nugget : float

    sill : float

    rang : float
           Semivariogram Range.

    Returns
    -------
    gamma : numpy array

    Notes
    -----
    Equation:

    (1) $\gamma = c0 + c * ( \frac{3}{2} * \frac{a}{h} - 0.5 * (\frac{a}{h})^{3} )$, $0 < a <= h$;
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
    [1] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.

    """
    ar = lags / rang
    a1 = (3/2) * ar
    a2 = 0.5 * ar**3

    gamma = np.where((lags <= rang),
                     (nugget + sill * (a1 - a2)),
                     (nugget + sill))
    g0 = gamma[0]
    gamma[0] = _get_zero_lag_value(lags[0], nugget, g0)
    return gamma
