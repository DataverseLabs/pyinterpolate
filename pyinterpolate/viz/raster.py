"""
Raster interpolation with ordinary kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import os
from typing import Dict, Tuple

import numpy as np
from libtiff import TIFFimage

from pyinterpolate.distance.point import point_distance
from pyinterpolate.kriging.point_kriging import kriging
from pyinterpolate.variogram.empirical.experimental_variogram import build_experimental_variogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


def set_dimensions(xs, ys, dmax):
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
                       direction=None,
                       tolerance=None,
                       allow_approx_solutions=True) -> Dict:
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

    direction : float (in range [0, 360]), optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y-axis and direction parameter. If ``tolerance`` is ``> 0``
        then the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance:

        * the major axis size == ``step_size``,
        * the minor axis size is ``tolerance * step_size``,
        * the baseline point is at a center of the ellipse,
        * the ``tolerance == 1`` creates an omnidirectional semivariogram.

    allow_approx_solutions : bool, default=True
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

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

    x_coords, y_coords, props = set_dimensions(data[:, 0], data[:, 1], dim)

    # Calculate semivariance if not provided

    if semivariogram_model is None:
        distances = point_distance(data[:, :-1], data[:, :-1])

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
                allow_approx_solutions=allow_approx_solutions)

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


def spatial_reference(pixel_size_x_direction: float,
                      pixel_size_y_direction: float,
                      x_origin: float,
                      y_origin: float) -> str:
    """
    Function creates content for ``.tfw`` file.

    Returns
    -------
    georeference : str
        Content of the file:

        - Line 1: A: pixel size in the x-direction in map units/pixel
        - Line 2: D: rotation about y-axis
        - Line 3: B: rotation about x-axis
        - Line 4: E: pixel size in the y-direction in map units, almost always negative2
        - Line 5: C: x-coordinate of the center of the upper left pixel
        - Line 6: F: y-coordinate of the center of the upper left pixel
        - All four parameters are expressed in the map units, which are described by the spatial reference system
          for the raster.

        Source: https://en.wikipedia.org/wiki/World_file
    """
    line_1 = str(pixel_size_x_direction) + '\n'
    line_2 = '0.0\n'
    line_3 = '0.0\n'
    line_4 = str(-1 * pixel_size_y_direction) + '\n'
    line_5 = str(x_origin) + '\n'
    line_6 = str(y_origin) + '\n'
    lines = [line_1, line_2, line_3, line_4, line_5, line_6]
    return ''.join(lines)


def to_tiff(data,
            dir_path: str,
            fname: str = '',
            dim=1000,
            number_of_neighbors=4,
            semivariogram_model=None,
            direction=None,
            tolerance=None,
            allow_approx_solutions=True) -> Tuple[str, str]:
    """
    Function interpolates raster from data points using ordinary kriging and stores output results in tiff and tfw
    files.

    Parameters
    ----------
    data : numpy array
        ``[coordinate x, coordinate y, value]``.

    dir_path : str
        Path to directory where output files will be stored.

    fname : str, default=''
        Suffix of the output ``*results.tiff`` and ``*error.tiff`` files.

    dim : int
        Number of pixels (points) of a larger dimension (it could be width or height). Ratio is preserved.

    number_of_neighbors : int, default=16
        Number of points used to interpolate data.

    semivariogram_model : TheoreticalVariogram, default=None
        Variogram model, if not provided then it is estimated from a given dataset.

    direction : float (in range [0, 360]), optional
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float (in range [0, 1]), optional
        If ``tolerance`` is 0 then points must be placed at a single line with the beginning in the origin of
        the coordinate system and the direction given by y-axis and direction parameter. If ``tolerance`` is ``> 0``
        then the bin is selected as an elliptical area with major axis pointed in the same direction as the line
        for 0 tolerance:

        * the major axis size == ``step_size``,
        * the minor axis size is ``tolerance * step_size``,
        * the baseline point is at a center of the ellipse,
        * the ``tolerance == 1`` creates an omnidirectional semivariogram.

    allow_approx_solutions : bool, default=True
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    files: Tuple[str, str]
        Tuple of two strings: path to tiff file with interpolated data and path to tiff file with interpolation errors.
    """

    results = interpolate_raster(data=data,
                                 dim=dim,
                                 number_of_neighbors=number_of_neighbors,
                                 semivariogram_model=semivariogram_model,
                                 direction=direction,
                                 tolerance=tolerance,
                                 allow_approx_solutions=allow_approx_solutions)

    interpolated_data = results['result']
    interpolation_errors = results['error']
    params = results['params']

    tfw_content = spatial_reference(
        pixel_size_x_direction=params['pixel size'],
        pixel_size_y_direction=params['pixel size'],
        x_origin=params['min x'],
        y_origin=params['max y']
    )

    data_tiff_fname, data_tfw_fname = _get_tiff_raster_fnames(dir_path, fname, 'results')
    error_tiff_fname, error_tfw_fname = _get_tiff_raster_fnames(dir_path, fname, 'error')

    # Save kriging results
    _write_files(tiff_fname=data_tiff_fname,
                 tfw_fname=data_tfw_fname,
                 data=interpolated_data,
                 tfw_content=tfw_content)

    # Save kriging errors
    _write_files(tiff_fname=error_tiff_fname,
                 tfw_fname=error_tfw_fname,
                 data=interpolation_errors,
                 tfw_content=tfw_content)

    return data_tiff_fname, error_tiff_fname


# Private functions of to_tiff function

def _get_tiff_raster_fnames(fpath: str, fname: str, ftype: str):
    """
    Function returns output file names for tiff files.

    Parameters
    ----------
    fpath : str
        Path to directory where output files will be stored.

    fname : str
        Suffix of the output ``*results.tiff`` and ``*error.tiff`` files.

    ftype : str
        File type, ``results`` or ``error``.

    Returns
    -------
    paths : Tuple[str, str]
        tiff file path and tfw file path.

    """
    if fname == '':
        tiff_fname = os.path.join(fpath, ftype + '.tiff')
        tfw_fname = os.path.join(fpath, ftype + '.tfw')
    else:
        tiff_fname = os.path.join(fpath, fname + '_' + ftype + '.tiff')
        tfw_fname = os.path.join(fpath, fname + '_' + ftype + '.tfw')

    return tiff_fname, tfw_fname


def _write_files(tiff_fname: str, tfw_fname: str, data: np.ndarray, tfw_content: str):
    """
    Function writes tiff file and tfw file.

    Parameters
    ----------
    tiff_fname : str
        Path to tiff file.

    tfw_fname : str
        Path to tfw file.

    data : numpy array
        Data to write to tiff file.

    tfw_content : str
        Content of tfw file.
    """

    _write_tiff(tiff_fname, data)
    _write_tfw(tfw_fname, tfw_content)


def _write_tiff(tiff_fname: str, data: np.ndarray):
    """
    Function writes tiff file and tfw file.

    Parameters
    ----------
    tiff_fname : str
        Path to tiff file.

    data : numpy array
        Data to write to tiff file.

    """
    npdata = data[..., np.newaxis]
    image = TIFFimage(data=npdata)
    image.write_file(tiff_fname)


def _write_tfw(tfw_fname: str, tfw_content: str):
    """
    Function writes tfw file.

    Parameters
    ----------
    tfw_fname : str
        Path to tfw file.

    tfw_content : str
        Content of tfw file.

    """
    with open(tfw_fname, 'w') as f:
        f.write(tfw_content)
