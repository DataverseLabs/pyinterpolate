from typing import Union, List, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from numpy import ndarray
from pandas import DataFrame, Series
from pydantic import field_validator, BaseModel, ConfigDict
from shapely.geometry import Point


class RawPoints(BaseModel):
    """
    Class represents points prepared for Experimental Variogram estimation
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    points: Union[List, Tuple, ndarray, GeoDataFrame, DataFrame]

    # noinspection PyNestedDecorators
    @field_validator('points')
    @classmethod
    def validate_dimensions(cls, v):
        if isinstance(v, List):
            if len(v[0]) != 3:
                raise ValueError('Points list must contain 3 values: '
                                 '[x, y, value]')
        elif isinstance(v, Tuple):
            if len(v[0]) != 3:
                raise ValueError('Points tuple must contain 3 values: '
                                 '[x, y, value]')
        elif isinstance(v, ndarray):
            if v.shape[1] != 3:
                raise ValueError('Points array must contain 3 values: '
                                 '[x, y, value]')
        elif isinstance(v, GeoDataFrame):
            if not len(v.columns) in (2, 3):
                raise ValueError('Passed GeoDataFrame must have 2 columns '
                                 '[geometry column, values column], or'
                                 ' 3 columns [x, y, values]')
        elif isinstance(v, DataFrame):
            if len(v.columns) != 3:
                raise ValueError('Points DataFrame must contain 3 columns '
                                 'representing x, y, value.')
        else:
            raise TypeError('Points must be list, tuple, numpy array,'
                            ' DataFrame, or GeoDataFrame')
        return v


class RawInterpolationPoints(BaseModel):
    """
    Class represents points prepared for Experimental Variogram estimation
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    points: Union[List, Tuple, ndarray, GeoDataFrame, DataFrame]

    # noinspection PyNestedDecorators
    @field_validator('points')
    @classmethod
    def validate_dimensions(cls, v):
        if isinstance(v, List):
            if len(v[0]) != 2:
                raise ValueError('Points list must contain 2 values: '
                                 '[x, y]')
        elif isinstance(v, Tuple):
            if len(v[0]) != 2:
                raise ValueError('Points tuple must contain 2 values: '
                                 '[x, y]')
        elif isinstance(v, ndarray):
            if v.shape[1] != 2:
                raise ValueError('Points array must contain 2 values: '
                                 '[x, y]')
        elif isinstance(v, GeoSeries):
            geom_types = v.geom_type.unique()
            if len(geom_types) != 1 or geom_types[0] != 'Point':
                raise ValueError('GeoSeries must contain only Point geometries')
        elif isinstance(v, DataFrame):
            if len(v.columns) != 2:
                raise ValueError('Points DataFrame must contain 2 columns '
                                 'representing x, y.')
        else:
            raise TypeError('Points must be list, tuple, numpy array,'
                            ' DataFrame, or GeoSeries')
        return v


class VariogramPoints:
    """
    Class represents points prepared for Experimental Variogram estimation
    and transformed to numpy array.
    """

    def __init__(self,
                 points: Union[
                     List, Tuple, ndarray, GeoDataFrame, GeoSeries, DataFrame
                 ]
                 ):

        # validate
        self.points = cast(RawPoints, points)

        # transform
        if not isinstance(self.points, ndarray):
            self.transform()

    def transform(self):
        """
        Method transform points to numpy array.
        """
        if isinstance(self.points, GeoDataFrame):
            cols = self.points.columns
            if len(cols) == 2:
                ds = self.points.copy()
                # geometry | values
                ds['x'] = ds[cols[0]].x
                ds['y'] = ds[cols[0]].y
                self.points = ds[['x', 'y', cols[1]]].values
            elif len(cols) == 3:
                self.points = self.points.values
        elif isinstance(self.points, DataFrame):
            self.points = self.points.values
        else:
            self.points = np.array(self.points)


class InterpolationPoints:
    """
    Class represents points prepared for interpolation.
    """

    def __init__(self,
                 points: Union[List, Tuple, ndarray, GeoSeries, Series, GeometryArray, ArrayLike]
                 ):
        # validate
        self.points = cast(RawInterpolationPoints, points)

        # transform
        if not isinstance(self.points, ndarray):
            self.transform()
        else:
            if self.points.ndim == 1:
                self.transform()

    def transform(self):
        """
        Method transform points to numpy array.
        """
        if isinstance(self.points, GeoSeries):
            xs = self.points.x
            ys = self.points.y
            self.points = np.column_stack((xs, ys))
        elif isinstance(self.points, DataFrame):
            self.points = self.points.values
        elif isinstance(self.points, Point):
            self.points = np.array([[self.points.x, self.points.y]])
        else:
            if np.array(self.points).ndim == 1:
                # if it's a single point, convert to 2D array
                self.points = np.array(self.points).reshape(1, -1)
            else:
                self.points = np.array(self.points)
