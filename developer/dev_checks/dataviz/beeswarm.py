from typing import Dict
import numpy as np

from pyinterpolate import VariogramCloud


def _get_bin_width(no, max_no, max_width=0.3):
    """
    Function gets bin width on a plot.

    Parameters
    ----------
    no : int
        The number of points within bin.

    max_no : int
        The maximum number of points within all bins.

    max_width : float
        The maximum deviation from the lag center in percent.

    Returns
    -------
    : float
        The deviation from the lag center.
    """

    bin_width = (no * max_width) / max_no
    return bin_width


def build_swarmplot_input(data: Dict, step_size: float, bins=None):
    """
    Function prepares data for lagged beeswarm plot.

    Parameters
    ----------
    data : Dict
        ``{lag: [values]}``

    step_size : float
        The step between lags.

    bins : int, optional
        If ``None`` given then number of bins per lag is chosen automatically.

    Returns
    -------
    : numpy array
        [lags (x-coordinates), values]
    """

    xs_arr = []
    ys_arr = []

    for lag, values in data.items():
        center = lag
        # bin values
        # TODO: set max points per level
        if bins is None:
            histogram, bin_edges = np.histogram(values, bins='auto')
        else:
            histogram, bin_edges = np.histogram(values, bins=bins)

        # Now prepare x-coordinates per lag
        x_indexes = []
        y_values = []

        # Define limits
        max_no = np.max(histogram)

        limits = [
            step_size * _get_bin_width(x, max_no) for x in histogram
        ]

        lower_limits = [center - l for l in limits]
        upper_limits = [center + l for l in limits]

        for idx, no_points in enumerate(histogram):
            xc = np.linspace(lower_limits[idx], upper_limits[idx], no_points)
            yc = [bin_edges[idx+1] for _ in xc]
            x_indexes.extend(xc)
            y_values.extend(yc)

        xs_arr.extend(x_indexes)
        ys_arr.extend(y_values)

    arr = np.array([xs_arr, ys_arr]).transpose()
    return arr


if __name__ == '__main__':
    import os

    def get_armstrong_data():
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.abspath(os.path.join(my_dir, filepath))
        arr = np.load(path_to_the_data)
        return arr

    ss = 1.5
    rng = 7
    _data = get_armstrong_data()
    vc = VariogramCloud(input_array=_data, step_size=ss, max_range=rng)
    ds = vc.experimental_point_cloud.copy()
    swarmplot_data = build_swarmplot_input(ds, step_size=ss, bins=10)
    print(swarmplot_data[0])
