from typing import Union

import numpy as np

from pyinterpolate.processing.structure import PolygonDataClass
from pyinterpolate.variogram.empirical.experimental_variogram import build_experimental_variogram


def build_areal_variogram(polyset: Union[dict, PolygonDataClass],
                          step_size: float,
                          max_range: float,
                          weights: np.ndarray = None,
                          direction: float = 0,
                          tolerance: float = 1):
    """
    Function prepares:
        - experimental semivariogram,
        - experimental covariogram,
        - variance.

    Parameters
    ----------
    polyset : Union[dict, PolygonDataClass]
              PolygonDataClass.polyset or PolygonDataClass object.

    step_size : float
                Distance between lags within each points are included in the calculations.

    max_range : float
                Maximum range of analysis.

    weights : numpy array or None, optional, default=None
              Weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted.

    direction : float (in range [0, 360]), optional, default=0
                direction of semivariogram, values from 0 to 360 degrees:
                    * 0 or 180: is NS direction,
                    * 90 or 270 is EW direction,
                    * 45 or 225 is NE-SW direction,
                    * 135 or 315 is NW-SE direction.

    tolerance : float (in range [0, 1]), optional, default=1
                If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                for 0 tolerance.
                    * The minor axis size is (tolerance * step_size)
                    * The major axis size is ((1 - tolerance) * step_size)
                    * The baseline point is at a center of the ellipse.
                Tolerance == 1 creates an omnidirectional semivariogram.

    Returns
    -------
    semivariogram_stats : EmpiricalSemivariogram
                          The class object with empirical semivariogram, empirical covariogram and variance

    See Also
    --------
    build_experimental_variogram : baseline method used by this function.
    EmpiricalSemivariogram : class that calculates and stores experimental semivariance, covariance and variance.

    Notes
    -----
    Function is an alias for build_experimental_variogram function and it reads centroids and values from
    PolygonDataClass.polyset attribute.

    Examples
    --------
    >>> geocls = PolygonDataClass()
    >>> geocls.from_file('testfile.shp', value_col='val', geometry_col='geometry', index_col='idx')
    >>> parsed = geocls.polyset
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = build_areal_variogram(parsed, step_size=STEP_SIZE, max_range=MAX_RANGE)
    """

    exp_variogram = build_experimental_variogram(polyset,
                                                 step_size=step_size,
                                                 max_range=max_range,
                                                 weights=weights,
                                                 direction=direction,
                                                 tolerance=tolerance)

    return exp_variogram
