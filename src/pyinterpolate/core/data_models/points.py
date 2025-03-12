from typing import Union, List, Tuple, cast

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from numpy import ndarray
from pandas import DataFrame
from pydantic import field_validator, BaseModel, ConfigDict


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
