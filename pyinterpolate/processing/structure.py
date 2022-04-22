from collections import OrderedDict
from typing import Union

import geopandas as gpd
import numpy as np

from pyinterpolate.processing.utils.exceptions import IndexColNotUniqueError, WrongGeometryTypeError


class PolygonDataClass:
    """
    Class stores and prepares aggregated data.

    Parameters
    ----------
    control_geometry : bool, default = True
                       Test if given geometries are all Polygon or MultiPolygon type.

    Attributes
    ----------
    polyset : dict
              Prepared dict with Polygon data.

    control_geometry : bool
                       Test if given geometries are all Polygon or MultiPolygon type.

    Methods
    -------.

    See Also
    --------
    Opt

    Notes
    -----
    Object structure:



    Examples
    --------

    """
    def __init__(self, control_geometry=True):
        self.control_geometry = control_geometry
        self.polyset = None

    def _check_index(self, ds, idx_col):
        dsl = len(ds)
        nuniq = ds[idx_col].nunique()
        if dsl != nuniq:
            raise IndexColNotUniqueError(dsl, nuniq)

    def _check_geometry(self, ds, geo_col):
        geometries = ds[geo_col].geom_type
        valid = ['Polygon', 'MultiPolygon']
        for gtype in geometries:
            if gtype not in valid:
                raise WrongGeometryTypeError(gtype)

    def _parse(self, dataset, val_col: str, idx_col: str, geo_col: str):
        cx = 'centroid.x'
        cy = 'centroid.x'

        # Build centroids
        centroids = dataset.centroid
        dataset[cx] = centroids.x
        dataset[cy] = centroids.y

        # Group data
        core_array = dataset[[cx, cy, val_col]].to_numpy()
        indexes_and_geometries = dataset[[idx_col, geo_col]].to_list()

        datadict = {
            'points': core_array,
            'igeom': indexes_and_geometries,
            'info': {
                'index_name': idx_col,
                'geom_name': geo_col,
                'val_name': val_col,
                'crs': dataset.crs
            }
        }

        return datadict

    def from_file(self,
                  fpath: str,
                  value_column: str,
                  geometry_col: str = 'geometry',
                  index_col: Union[str, None] = None):
        """
        Loads areal dataset from a file supported by GeoPandas.

        Parameters
        ----------
        fpath : str
                Path to the spatial file.

        value_column : str
                       The name of a column with values.

        geometry_col : str, default='geometry'
                       The name of a column with polygons.

        index_col : str or None, default = None
                    Index column name. It could be any unique value from a dataset. If not given then index is created
                    by an algorithm as a set of values in range 0:length of a dataset.

        Returns
        -------


        Raises
        ------
        IndexColNotUniqueError : Raised when given index column is not unique.

        WrongGeometryTypeError : Raised if given geometry is different than Polygon or MultiPolygon.

        """

        dataset = gpd.read_file(fpath)

        if index_col is not None:
            dataset = dataset[[index_col, geometry_col, value_column]]
            # Check indices
            self._check_index(dataset, index_col)
        else:
            index_col = 'index'
            dataset = dataset[[geometry_col, value_column]]
            dataset[index_col] = np.arange(0, len(dataset))
            dataset = dataset[[index_col, geometry_col, value_column]]

        # Check geometries
        self._check_geometry(dataset, geometry_col)

        # Update polyset
        self.polyset = self._parse(dataset, val_col=value_column, idx_col=index_col, geo_col=geometry_col)

    def from_geodataframe(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass
    

def get_polyset():
    pass