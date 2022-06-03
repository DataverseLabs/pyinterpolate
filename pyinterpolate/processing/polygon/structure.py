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
              Prepared dict with Polygon data. It's structure is:
              polyset = {
                'points': numpy array with centroid.x, centroid.y and value,
                'igeom': list of [index, geometry polygon],
                'info': {
                    'index_name': the name of the index column,
                    'geom_name': the name of the geometry column,
                    'val_name': the name of the value column,
                    'crs': CRS of a dataset
                }
            }

    control_geometry : bool
                       Test if given geometries are all Polygon or MultiPolygon type.

    Methods
    -------
    from_file(fpath, value_col, geometry_col, index_col)
        Reads and parses data from spatial file supported by GeoPandas.

    from_geodataframe(gdf, value_col, geometry_col, use_index)
        Reads and parses data from GeoPandas GeoDataFrame.

    Examples
    --------
    >>> geocls = PolygonDataClass()
    >>> geocls.from_file('testfile.shp', value_col='val', geometry_col='geometry', index_col='idx')
    >>> parsed = geocls.polyset
    >>> print(list(parsed.keys()))
    (points, igeom, info)

    """
    def __init__(self, control_geometry=True):
        self.control_geometry = control_geometry
        self.polyset = None

    @staticmethod
    def _check_index(ds, idx_col):
        dsl = len(ds)
        nuniq = ds[idx_col].nunique()
        if dsl != nuniq:
            raise IndexColNotUniqueError(dsl, nuniq)

    @staticmethod
    def _check_geometry(ds, geo_col):
        geometries = ds[geo_col].geom_type
        valid = ['Polygon', 'MultiPolygon']
        for gtype in geometries:
            if gtype not in valid:
                raise WrongGeometryTypeError(gtype)

    @staticmethod
    def _parse(dataset, val_col: str, geo_col: str, idx_col: str):
        cx = 'centroid.x'
        cy = 'centroid.x'

        # Build centroids
        centroids = dataset.centroid
        dataset[cx] = centroids.x
        dataset[cy] = centroids.y

        # Group data
        core_array = dataset[[cx, cy, val_col]].to_numpy()
        indexes_and_geometries = dataset[[idx_col, geo_col]].to_numpy()

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
                  value_col: str,
                  geometry_col: str = 'geometry',
                  index_col: Union[str, None] = None) -> None:
        """
        Loads areal dataset from a file supported by GeoPandas.

        Parameters
        ----------
        fpath : str
                Path to the spatial file.

        value_col : str
                    The name of a column with values.

        geometry_col : str, default='geometry'
                       The name of a column with polygons.

        index_col : str or None, default = None
                    Index column name. It could be any unique value from a dataset. If not given then index is created
                    by an algorithm as a set of values in range 0:length of a dataset.


        Raises
        ------
        IndexColNotUniqueError : Raised when given index column is not unique.

        WrongGeometryTypeError : Raised if given geometry is different than Polygon or MultiPolygon.

        """

        dataset = gpd.read_file(fpath)

        if index_col is not None:
            dataset = dataset[[index_col, geometry_col, value_col]]
            # Check indices
            self._check_index(dataset, index_col)
        else:
            index_col = 'index'
            dataset = dataset[[geometry_col, value_col]]
            dataset[index_col] = np.arange(0, len(dataset))
            dataset = dataset[[index_col, geometry_col, value_col]]

        # Check geometries
        self._check_geometry(dataset, geometry_col)

        # Update polyset
        self.polyset = self._parse(dataset, val_col=value_col, geo_col=geometry_col, idx_col=index_col)

    def from_geodataframe(self,
                          gdf: gpd.GeoDataFrame,
                          value_col: str,
                          geometry_col: str = 'geometry',
                          use_index: bool = True,
                          index_col: Union[str, None] =  None):
        """
        Loads areal dataset from a GeoDataFrame supported by GeoPandas.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame

        value_col : str
                    The name of a column with values.

        geometry_col : str, default = 'geometry'
                       The name of a column with polygons.

        use_index : bool, default = True
                    Should the GeoDataFrame index be used as a column.

        index_col : Union[str, None], default = None
                    If set then a specific column is treated as an index.

        Raises
        ------
        WrongGeometryTypeError : Raised if given geometry is different than Polygon or MultiPolygon.

        """

        # Check geometries
        self._check_geometry(gdf, geometry_col)

        dataset = gdf[[value_col, geometry_col]]

        if index_col is not None:
            self._check_index(dataset, index_col)
            idx_name = index_col
        else:
            if use_index:
                idx_name = dataset.index.name

                if idx_name is None:
                    dataset.index.name = 'index'
                idx_name = 'index'

                dataset.reset_index(inplace=True)

            else:
                idx_name = 'index'
                dataset[idx_name] = np.arange(0, len(dataset))

        self.polyset = self._parse(dataset,
                                   val_col=value_col,
                                   geo_col=geometry_col,
                                   idx_col=idx_name)


def get_polyset_from_geodataframe(gdf: gpd.GeoDataFrame,
                                  value_col: str,
                                  geometry_col: str = 'geometry',
                                  use_index=True,
                                  index_col: Union[str, None] =  None):
    """
    Function prepares polyset object from GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame

    value_col : str
                Value column (aggregated value).

    geometry_col : str, default = 'geometry'
                   Column with geometry.

    use_index : bool, default = True
                Uses passed GeoDataFrame index as area indices.

    index_col : Union[str, None], default = None
                    If set then a specific column is treated as an index.

    Returns
    -------
    polyset : dict
              Prepared dict with Polygon data. It's structure is:
              polyset = {
                'points': numpy array with centroid.x, centroid.y and value,
                'igeom': list of [index, geometry polygon],
                'info': {
                    'index_name': the name of the index column,
                    'geom_name': the name of the geometry column,
                    'val_name': the name of the value column,
                    'crs': CRS of a dataset
                }
            }

    """
    polyclass = PolygonDataClass()
    polyclass.from_geodataframe(gdf, value_col, geometry_col, use_index)
    return polyclass.polyset


def get_polyset_from_file(fpath: str,
                          value_col: str,
                          geometry_col: str = 'geometry',
                          index_col: Union[str, None] = None):
    """
    Function prepares polyset object from spatial file.

    Parameters
    ----------
    fpath : str
            Path to the file.

    value_col : str
                Value column (aggregated value).

    geometry_col : str, default = 'geometry'
                   Column with geometry.

    index_col : str, default = None
                Index values column name, default is None and index is created as a set of number in range 0 to number
                of areas.

    Returns
    -------
    polyset : dict
              Prepared dict with Polygon data. It's structure is:
              polyset = {
                'points': numpy array with centroid.x, centroid.y and value,
                'igeom': list of [index, geometry polygon],
                'info': {
                    'index_name': the name of the index column,
                    'geom_name': the name of the geometry column,
                    'val_name': the name of the value column,
                    'crs': CRS of a dataset
                }
            }

    """
    polyclass = PolygonDataClass()
    polyclass.from_file(fpath, value_col, geometry_col, index_col)
    return polyclass.polyset
