"""
Raster interpolation with ordinary kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict

import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.variogram.empirical.experimental_variogram import build_experimental_variogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram
from pyinterpolate.kriging.point_kriging import kriging


def _set_dims(xs, ys, dmax):
    """
    Function sets dimensions of the output array.

    Parameters
    ----------
    xs : numpy array
         X coordinates.

    ys : numpy array
         Y coordinates.

    dmax : int
           How many points between max dimensions.

    Returns
    -------
    : List
        x_dim_coords, y_dim_coords, [properties]
    """

    xmin = np.min(xs)
    xmax = np.max(xs)

    ymin = np.min(ys)
    ymax = np.max(ys)

    x_abs = np.abs(xmax - xmin)
    y_abs = np.abs(ymax - ymin)

    if x_abs > y_abs:
        step = x_abs / dmax
        x_dim_coords = np.arange(xmin + step, xmax + step, step)
        y_dim_coords = np.arange(ymin + step, ymax + step, step)
    else:
        step = y_abs / dmax
        y_dim_coords = np.arange(ymin + step, ymax + step, step)
        x_dim_coords = np.arange(xmin + step, xmax + step, step)

    # y_dim_coords must be flipped

    y_dim_coords = y_dim_coords[::-1]
    return x_dim_coords, y_dim_coords, [step, xmin, xmax, ymin, ymax]


def interpolate_raster(data,
                       dim=1000,
                       number_of_neighbors=4,
                       semivariogram_model=None,
                       direction=0,
                       tolerance=1) -> Dict:
    """
    Function interpolates raster from data points using ordinary kriging.

    Parameters
    ----------
    data : numpy array
        ``[coordinate x, coordinate y, value]``.

    dim : int
        Number of pixels (points) of a larger dimension (it could be width or height). Ratio is preserved.

    number_of_neighbors : int, default=16
        Number of points used to interpolate data.

    semivariogram_model : TheoreticalVariogram, default=None
        Variogram model, if not provided then it is estimated from a given dataset.

    direction : float (in range [0, 360]), default = 0
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional, default=1
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y axis and direction parameter. If ``tolerance`` is ``> 0`` then
        the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance:

        * the major axis size == ``step_size``,
        * the minor axis size is ``tolerance * step_size``,
        * the baseline point is at a center of the ellipse,
        * the ``tolerance == 1`` creates an omnidirectional semivariogram.

    Returns
    -------
    raster_dict : Dict
        A dictionary with keys:

        * **'result'**: numpy array of interpolated values,
        * **'error'**: numpy array of interpolation errors,
        * **'params'**:
            * 'pixel size',
            * 'min x',
            * 'max x',
            * 'min y',
            * 'max y'

    """

    # Set dimension

    if isinstance(data, list):
        data = np.array(data)

    x_coords, y_coords, props = _set_dims(data[:, 0], data[:, 1], dim)

    # Calculate semivariance if not provided

    if semivariogram_model is None:
        distances = calc_point_to_point_distance(data[:, :-1])

        maximum_range = np.max(distances)
        number_of_divisions = 100
        step_size = maximum_range / number_of_divisions

        evariogram = build_experimental_variogram(input_array=data,
                                                  step_size=step_size,
                                                  max_range=maximum_range,
                                                  direction=direction,
                                                  tolerance=tolerance)

        ts = TheoreticalVariogram()
        ts.autofit(experimental_variogram=evariogram)
    else:
        ts = semivariogram_model

    # Interpolate data point by point
    interpolation_points = []

    for ridx, _y in enumerate(y_coords):
        for cidx, _x in enumerate(x_coords):
            coords = np.array([_x, _y])
            interpolation_points.append(coords)

    k = kriging(observations=data,
                theoretical_model=ts,
                points=interpolation_points,
                how='ok',
                no_neighbors=number_of_neighbors,
                err_to_nan=True)

    kriged_matrix = k[:, 0].reshape((len(y_coords), len(x_coords)))
    kriged_errors = k[:, 1].reshape((len(y_coords), len(x_coords)))

    raster_dict = {
        'result': kriged_matrix,
        'error': kriged_errors,
        'params': {
            'pixel size': props[0],
            'min x': props[1],
            'max x': props[2],
            'min y': props[3],
            'max y': props[4]
        }
    }

    return raster_dict
