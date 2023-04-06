"""
Core data structures for block interpolation.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import logging
from typing import Union

import geopandas as gpd
import pandas as pd

from pyinterpolate.processing.utils.exceptions import IndexColNotUniqueError


class Blocks:
    """Class stores and prepares aggregated data.

    Attributes
    ----------
    data : gpd.GeoDataFrame
        Dataset with block values.

    value_column_name : Any
        Name of the column with block rates.

    geometry_column_name : Any
        Name of the column with a block geometry.

    index_column_name : Any
        Name of the column with the index.

    Methods
    -------
    from_file()
        Reads and parses data from spatial file supported by GeoPandas.

    from_geodataframe()
        Reads and parses data from GeoPandas ``GeoDataFrame``.

    Examples
    --------
    >>> geocls = Blocks()
    >>> geocls.from_file('testfile.shp', value_col='val', geometry_col='geometry', index_col='idx')
    >>> parsed_columns = geocls.data.columns
    >>> print(list(parsed_columns))
    (idx, val, geometry, centroid.x, centroid.y)

    """

    def __init__(self):
        self.data = None
        self.value_column_name = None
        self.index_column_name = None
        self.geometry_column_name = None
        self.cx = 'centroid_x'
        self.cy = 'centroid_y'

    @staticmethod
    def _check_index(ds, idx_col):
        dsl = len(ds)
        nuniq = ds[idx_col].nunique()
        if dsl != nuniq:
            raise IndexColNotUniqueError(dsl, nuniq)

    def _parse(self, dataset, val_col, geo_col, idx_col):
        """Parser.

        Parameters
        ----------
        dataset : gpd.GeoDataFrame
                  GeoDataFrame with selected index, value and geometry columns and calculated centroid x and
                  centroid y coordinates.

        val_col : Any
                  Name of the column with block rates.

        geo_col : Any
                  Name of the column with a block geometry.

        idx_col : Any
                  Name of the column with the index.

        Returns
        -------
        dataset : gpd.GeoDataFrame
                  dataset[[idx_col, geo_col, val_col, self.cx, self.cy]]
        """

        pd.options.mode.chained_assignment = None  # Disable setting with copy warning

        # Build centroids
        centroids = dataset[geo_col].centroid
        cxs = [float(c) for c in centroids.x]
        cys = [float(c) for c in centroids.y]
        dataset[self.cx] = cxs
        dataset[self.cy] = cys

        return dataset[[idx_col, geo_col, val_col, self.cx, self.cy]]

    def set_names(self, val_col, geom_col, idx_col):
        self.value_column_name = val_col
        self.index_column_name = idx_col
        self.geometry_column_name = geom_col

    def from_file(self,
                  fpath: str,
                  value_col,
                  geometry_col='geometry',
                  index_col=None,
                  layer_name=None):
        """
        Loads areal dataset from a file supported by GeoPandas.

        Parameters
        ----------
        fpath : str
            Path to the spatial file.

        value_col : Any
            The name of a column with values.

        geometry_col : default='geometry'
            The name of a column with blocks.

        index_col : default = None
            Index column name. It could be any unique value from a dataset. If not given then index is taken
            from the index array of ``GeoDataFrame``, and it is named ``'index'``.

        layer_name : Any, default = None
            The name of a layer with data if provided input is a *gpkg* file.

        Raises
        ------
        IndexColNotUniqueError
            Raised when given index column has not unique values.

        """

        if fpath.lower().endswith('.gpkg'):
            dataset = gpd.read_file(fpath, layer=layer_name)
        else:
            dataset = gpd.read_file(fpath)

        if index_col is not None:
            dataset = dataset[[index_col, geometry_col, value_col]]
            # Check indices
            self._check_index(dataset, index_col)
        else:
            if dataset.index.name:
                index_col = dataset.index.name
                dataset = dataset.reset_index()
            else:
                index_col = 'index'
                dataset.index.name = index_col
                dataset = dataset.reset_index()

        # Update polyset
        self.data = self._parse(dataset, val_col=value_col, geo_col=geometry_col, idx_col=index_col)
        self.set_names(val_col=value_col, geom_col=geometry_col, idx_col=index_col)

    def from_geodataframe(self,
                          gdf: gpd.GeoDataFrame,
                          value_col,
                          geometry_col='geometry',
                          index_col=None):
        """
        Loads areal dataset from a GeoDataFrame supported by GeoPandas.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame

        value_col : Any
            The name of a column with values.

        geometry_col : Any, default = 'geometry'
            The name of a column with blocks.

        index_col : Any, default = None
            If set then a specific column is treated as an index.

        Raises
        ------
        IndexColNotUniqueError
            Given index column values are not unique.

        """
        if index_col is not None:
            self._check_index(gdf, index_col)
            idx_name = index_col
        else:
            if gdf.index.name:
                idx_name = gdf.index.name
                gdf = gdf.reset_index()
            else:
                idx_name = 'index'
                gdf.index.name = idx_name
                gdf = gdf.reset_index()

        self.data = self._parse(gdf,
                                val_col=value_col,
                                geo_col=geometry_col,
                                idx_col=idx_name)
        self.set_names(val_col=value_col, geom_col=geometry_col, idx_col=idx_name)


class PointSupport:
    """Class prepares the point support data in relation to block dataset.

    Parameters
    ----------
    log_not_used_points : bool, default=False
        Should dropped points be logged?

    Attributes
    ----------
    point_support : gpd.GeoDataFrame
        Dataset with point support values and indexes of blocks (where points fall into).

    value_column : str
        The value column name

    geometry_column : str
        The geometry column name.

    block_index_column : str
        The area index.

    x_col : str, default = "x_col"
        Longitude column name.

    y_col : str, default = "y_col"
        Latitude column name.

    log_dropped : bool
        See log_not_used_points parameter.

    Methods
    -------
    from_files()
        Loads point support and polygon data from files.

    from_geodataframes()
        Loads point support and polygon data from dataframe.

    Notes
    -----
    The PointSupport class structure is designed to store the information about the points within polygons.
    During the regularization process, the inblock variograms are estimated from the polygon's point support, and
    semivariances are calculated between point supports of neighbouring blocks.

    The class takes population grid (support) and blocks data (polygons). Then, spatial join is performed and points
    are assigned to areas within they are placed. The core attribute is ``point_support`` - GeoDataFrame with columns:

    * ``x_col`` - a floating representation of longitude,
    * ``y_col`` - a floating representation of latitude,
    * ``value_column`` - the attribute which describes the name of a column with the point-support's value,
    * ``geometry_column`` - the attribute which describes the name of a geometry column with ``Point()`` representation
      of the point support coordinates,
    * ``block_index_column`` - the name of a column which directs to the block index values.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from pyinterpolate import PointSupport
    >>>
    >>>
    >>> POPULATION_DATA = "path to the point support file"
    >>> POLYGON_DATA = "path to the polygon data"
    >>> GEOMETRY_COL = "geometry"
    >>> POP10 = "POP10"
    >>> POLYGON_ID = "FIPS"
    >>>
    >>> gdf_points = gpd.read_file(POPULATION_DATA)
    >>> gdf_polygons = gpd.read_file(POLYGON_DATA)
    >>> point_support = PointSupport()
    >>> out = point_support.from_geodataframes(gdf_points,
    ...                                        gdf_polygons,
    ...                                        point_support_geometry_col=GEOMETRY_COL,
    ...                                        point_support_val_col=POP10,
    ...                                        blocks_geometry_col=GEOMETRY_COL,
    ...                                        blocks_index_col=POLYGON_ID)
    """

    def __init__(self, log_not_used_points=False):
        self.point_support = None
        self.value_column = None
        self.geometry_column = None
        self.block_index_column = None
        self.x_col = 'x_col'
        self.y_col = 'y_col'
        self.log_dropped = log_not_used_points

    def from_files(self,
                   point_support_data_file: str,
                   blocks_file: str,
                   point_support_geometry_col,
                   point_support_val_col,
                   blocks_geometry_col,
                   blocks_index_col,
                   use_point_support_crs: bool = True,
                   point_support_layer_name=None,
                   blocks_layer_name=None):
        """
        Methods prepares the point support data from files.

        Parameters
        ----------
        point_support_data_file : str
            Path to the file with point support data. Reads all files processed by the GeoPandas.

        blocks_file : str
            Path to the file with polygon data. Reads all files processed by GeoPandas.

        point_support_geometry_col : Any
            The name of the point support geometry column.

        point_support_val_col : Any
            The name of the point support column with values.

        blocks_geometry_col : Any
            The name of the polygon geometry column.

        blocks_index_col : Any
            The name of polygon's index column (must be unique!).

        use_point_support_crs : bool, default = True
            If set to ``False`` then the point support crs is transformed to the same crs as polygon dataset.

        point_support_layer_name : Any, default = None
            If provided file is *.gpkg* then this parameter must be provided.

        blocks_layer_name  : Any, default = None
            If provided file is *.gpkg* then this parameter must be provided.
        """
        # Load data
        if point_support_data_file.lower().endswith('.gpkg'):
            point_support = gpd.read_file(point_support_data_file, layer=point_support_layer_name)
        else:
            point_support = gpd.read_file(point_support_data_file)

        if blocks_file.lower().endswith('.gpkg'):
            blocks = gpd.read_file(blocks_file, layer=blocks_layer_name)
        else:
            blocks = gpd.read_file(blocks_file)

        self.from_geodataframes(point_support,
                                blocks,
                                point_support_val_col=point_support_val_col,
                                blocks_geometry_col=blocks_geometry_col,
                                blocks_index_col=blocks_index_col,
                                point_support_geometry_col=point_support_geometry_col,
                                use_point_support_crs=use_point_support_crs)

    def from_geodataframes(self,
                           point_support_dataframe: Union[gpd.GeoDataFrame, gpd.GeoSeries],
                           blocks_dataframe: gpd.GeoDataFrame,
                           point_support_geometry_col,
                           point_support_val_col,
                           blocks_geometry_col,
                           blocks_index_col,
                           use_point_support_crs: bool = True):
        """
        Methods prepares the point support data from files.

        Parameters
        ----------
        point_support_dataframe : GeoDataFrame or GeoSeries

        blocks_dataframe : GeoDataFrame
            Block data with block indexes and geometries.

        point_support_geometry_col : Any
            The name of the point support geometry column.

        point_support_val_col : Any
            The name of the point support column with values.

        point_support_val_col : Any
            The name of the point support column with values.

        blocks_geometry_col : Any
            The name of the polygon geometry column.

        blocks_index_col : Any
            The name of polygon's index column (must be unique!).

        use_point_support_crs : bool, default = True
            If set to ``False`` then the point support crs is transformed to the same crs as polygon dataset.
        """
        # Select data
        point_support, blocks = self._select_data(point_support_dataframe,
                                                  blocks_dataframe,
                                                  point_support_geometry_col,
                                                  point_support_val_col,
                                                  blocks_geometry_col,
                                                  blocks_index_col)

        # Transform CRS
        point_support, blocks = self._transform_crs(point_support, blocks, use_point_support_crs)

        # Merge data
        joined = gpd.sjoin(point_support, blocks, how='left')

        # Check which points weren't joined
        if self.log_dropped:
            is_na = joined.isna().any(axis=1)
            not_joined_points = joined[is_na]['geometry']
            if len(not_joined_points) > 0:
                logging.info('POINT SUPPORT : Dropped points:')
                for pt in not_joined_points:
                    msg = '({}, {})'.format(pt.x, pt.y)
                    logging.info(msg)

        # Clean data
        joined.dropna(inplace=True)
        joined = joined[['index_right', point_support_geometry_col, point_support_val_col]]
        joined = self._set_dtypes(joined,
                                  blocks,
                                  point_support_val_col,
                                  'index_right')

        if blocks_index_col is not None and blocks_index_col != 'index_right':
            joined.rename(columns={'index_right': blocks_index_col}, inplace=True)

        # Set x, y coordinates
        joined[self.x_col] = [float(c) for c in joined[point_support_geometry_col].x]
        joined[self.y_col] = [float(c) for c in joined[point_support_geometry_col].y]

        # Set attributes
        self.point_support = joined
        self.geometry_column = point_support_geometry_col
        self.value_column = point_support_val_col
        self.block_index_column = blocks_index_col

    def _select_data(self,
                     point_support: gpd.GeoDataFrame,
                     blocks: gpd.GeoDataFrame,
                     point_support_geometry_col,
                     point_support_value_col,
                     blocks_geometry_col,
                     blocks_index_col):

        point_support = self._select_point_support(point_support,
                                                   point_support_geometry_col,
                                                   point_support_value_col)
        blocks = self._select_blocks(blocks,
                                     blocks_geometry_col,
                                     blocks_index_col)
        return point_support, blocks

    @staticmethod
    def _select_point_support(point_support: gpd.GeoDataFrame,
                              geometry_col,
                              value_col):
        point_support = point_support[[geometry_col, value_col]]
        return point_support

    @staticmethod
    def _select_blocks(polygon_data: gpd.GeoDataFrame,
                       geometry_col,
                       index_col):
        polygon_data = polygon_data[[geometry_col, index_col]]
        polygon_data.set_index(index_col, inplace=True)
        return polygon_data

    @staticmethod
    def _transform_crs(points: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, use_points: bool):
        if use_points:
            if blocks.crs != points.crs:
                blocks = blocks.to_crs(points.crs)
        else:
            if blocks.crs != points.crs:
                points = points.to_crs(blocks.crs)
        return points, blocks

    @staticmethod
    def _set_dtypes(joined, blocks, point_support_val_col, blocks_id):
        index_dtype = blocks.index.dtype

        joined[blocks_id] = joined[blocks_id].astype(index_dtype)
        joined[point_support_val_col] = joined[point_support_val_col].astype(float)
        return joined
