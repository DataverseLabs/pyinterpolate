from typing import Union, Dict, Type
from numpy.typing import ArrayLike

from pyinterpolate.core.data_models.points import VariogramPoints
from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram


class DirectionalVariogram:
    """
    Class prepares four directional variograms and isotropic variogram.

    Parameters
    ----------
    ds : ArrayLike, optional
        ``[x, y, value]``

    values : ArrayLike, optional
        Observation in the i-th geometry (from ``geometries``). Optional
        parameter, if not given then ``ds`` must be provided.

    geometries : ArrayLike, optional
        Array or similar structure with geometries. It must have the same
        length as ``values``. Optional parameter, if not given then ``ds``
        must be provided. Point type geometry.

    step_size : float
        The fixed distance between lags grouping point neighbors.

    max_range : float
        The maximum distance at which the semivariance is calculated.

    tolerance : float, optional
        If ``tolerance`` is 0 then points must be placed at a single line with
        the beginning in the origin of the coordinate system and the
        direction given by y-axis and direction parameter.
        If ``tolerance`` is ``> 0`` then the bin is selected as an elliptical
        area with major axis pointed in the same direction as the line for
        ``0`` tolerance.

        * The major axis size == ``step_size``.
        * The minor axis size is ``tolerance * step_size``
        * The baseline point is at a center of the ellipse.
        * The ``tolerance == 1`` creates an omnidirectional semivariogram.

    custom_bins : numpy array, optional
        Custom bins for semivariance calculation. If provided, then parameter
        ``step_size`` is ignored and ``max_range`` is set to the final bin
        distance.

    custom_weights : numpy array, optional
        Custom weights assigned to points. Only semivariance values are
        weighted.

    Attributes
    ----------
    directional_variograms : Dict
        Dictionary with five variograms:

        * ``ISO``: isotropic,
        * ``NS``: North-South axis,
        * ``WE``: West-East axis,
        * ``NE-SW``: Northeastern-Southwestern axis,
        * ``NW-SE``: Northwestern-Southeastern axis.

    directions : Dict
        Dictionary where keys are directions: NS, WE, NE-SW, NW-SE, and
        values are angles: 90, 0, 45, 135

    ds : numpy array
        See ``ds`` parameter.

    step_size : float
        See ``step_size`` parameter.

    max_range : float
        See ``max_range`` parameter.

    tolerance : float
        See ``tolerance`` parameter.

    custom_bins : numpy array, optional
        See ``custom_bins`` parameter.

    custom_weights : float
        See ``custom_weights`` parameter.

    Methods
    -------
    get()
        Returns copy of calculated directional variograms or
        single variogram in a specific direction.

    show()
        Plot all variograms.

    Examples
    --------
    >>> import numpy as np
    >>> from pyinterpolate import DirectionalVariogram
    >>>
    >>>
    >>> ds = np.load('armstrong_data.npy')
    >>> step_size = 1.5
    >>> max_range = 6
    >>> directional_variogram = DirectionalVariogram(
    ...     step_size=STEP_SIZE,
    ...     max_range=MAX_RANGE,
    ...     values=ds[:, -1],
    ...     geometries=ds[:, :-1]
    ... )
    >>> variograms = directional_variogram.get()
    >>> print(variograms.keys())
    dict_keys(['ISO', 'NS', 'WE', 'NE-SW', 'NW-SE'])
    >>> directional_variogram.show()  # shows variograms in all directions
    """

    def __init__(self,
                 step_size: float,
                 max_range: float,
                 ds: Union[ArrayLike, VariogramPoints] = None,
                 values: ArrayLike = None,
                 geometries: ArrayLike = None,
                 tolerance: float = 0.2,
                 custom_weights=None,
                 custom_bins=None):

        # Validate points
        if not isinstance(ds, VariogramPoints):
            ds = VariogramPoints(points=ds,
                                 geometries=geometries,
                                 values=values)
            ds = ds.points

        self.ds = ds
        self.custom_bins = custom_bins

        self.step_size = step_size
        self.max_range = max_range

        self.tolerance = tolerance
        self.custom_weights = custom_weights
        self.possible_variograms = ['ISO', 'NS', 'WE', 'NE-SW', 'NW-SE']
        self.directions = {
            'NS': 90,
            'WE': 0,
            'NE-SW': 45,
            'NW-SE': 135
        }
        self.directional_variograms = {}
        self._build_experimental_variograms()

    def _build_experimental_variograms(self):
        isotropic = ExperimentalVariogram(
            ds=self.ds,
            step_size=self.step_size,
            max_range=self.max_range,
            custom_bins=self.custom_bins,
            custom_weights=self.custom_weights)

        self.directional_variograms['ISO'] = isotropic

        for idx, val in self.directions.items():
            variogram = ExperimentalVariogram(
                ds=self.ds,
                step_size=self.step_size,
                max_range=self.max_range,
                custom_bins=self.custom_bins,
                custom_weights=self.custom_weights,
                direction=val,
                tolerance=self.tolerance)

            self.directional_variograms[idx] = variogram

    def get(self, direction=None) -> Union[
        Dict, Type["ExperimentalVariogram"]
    ]:
        """
        Method returns all variograms or a single variogram in a specific
        direction.

        Parameters
        ----------
        direction : str, default = None
            The direction of variogram from a list of ``possible_variograms``
            attribute: "ISO", "NS", "WE", "NE-SW", "NW-SE".

        Returns
        -------
        : Union[Dict, Type[ExperimentalVariogram]]
            The dictionary with variograms for all possible directions,
            or a single variogram for a specific direction.
        """
        if direction is None:
            return self.directional_variograms.copy()
        else:
            if direction in self.possible_variograms:
                return self.directional_variograms[direction]

            msg = (f'Given direction is not possible to retrieve, '
                   f'pass one direction from a possible_variograms: '
                   f'{self.possible_variograms} or leave ``None`` to get '
                   f'a dictionary with all possible variograms.')
            raise KeyError(msg)

    def show(self):
        """
        Method shows variograms in all directions.
        """
        import matplotlib.pyplot as plt

        if self.directional_variograms:
            _lags = self.directional_variograms['ISO'].lags
            _ns = self.directional_variograms['NS'].semivariances
            _we = self.directional_variograms['WE'].semivariances
            _nw_se = self.directional_variograms[
                'NW-SE'].semivariances
            _ne_sw = self.directional_variograms[
                'NE-SW'].semivariances
            _iso = self.directional_variograms['ISO'].semivariances

            plt.figure(figsize=(20, 8))
            plt.plot(_lags, _iso, color='#1b9e77')
            plt.plot(_lags, _ns, '--', color='#d95f02')
            plt.plot(_lags, _we, '--', color='#7570b3')
            plt.plot(_lags, _nw_se, '--', color='#e7298a')
            plt.plot(_lags, _ne_sw, '--', color='#66a61e')
            plt.title('Comparison of experimental semivariance models')
            plt.legend(['Isotropic',
                        'NS',
                        'WE',
                        'NW-SE',
                        'NE-SW'])
            plt.xlabel('Distance')
            plt.ylabel('Semivariance')
            plt.show()
