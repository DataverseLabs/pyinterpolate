from typing import Union, Dict

import geopandas as gpd
import pandas as pd


class PointSupportDataClass:
    """
    Class stores and describes the point support data in relation to polygon dataset.

    Attributes
    ----------
    pointset : Dict
               Prepared dict with Polygon data. It's structure is:
               polyset = {
                   'data': {id: [[point support array - lon, lat, value]], },
                   'info': {
                       'crs': CRS of a dataset
                   }
               }

    Methods
    -------
    from_files()
        Loads point support and polygon data from files.

    from_geodataframes()
        Loads point support and polygon data from dataframe.

    Notes
    -----
    TODO: info about regualarization process

    Examples
    --------
    import geopandas as gpd
    from pyinterpolate.processing.point.structure import PointSupportDataClass


    POPULATION_DATA = "path to the point support file"
    POLYGON_DATA = "path to the polygon data"
    GEOMETRY_COL = "geometry"
    POP10 = "POP10"
    POLYGON_ID = "FIPS"

    gdf_points = gpd.read_file(POPULATION_DATA)
    gdf_polygons = gpd.read_file(POLYGON_DATA)
    point_support = PointSupportDataClass()
    out = point_support.from_geodataframes(gdf_points,
                                           gdf_polygons,
                                           point_support_geometry_col=GEOMETRY_COL,
                                           point_support_val_col=POP10,
                                           polygon_geometry_col=GEOMETRY_COL,
                                           polygon_index_col=POLYGON_ID)
    """

    def __init__(self):
        self.pointset = None

    def from_files(self,
                   point_support_data_file: str,
                   polygon_file: str,
                   point_support_geometry_col: str,
                   point_support_val_col: str,
                   polygon_geometry_col: str,
                   polygon_index_col: str,
                   use_point_support_crs: bool = True,
                   dropna: bool = True,
                   point_support_layer_name: str = None,
                   polygon_layer_name: str = None):
        """
        Methods prepares the point support data from files.

        Parameters
        ----------
        point_support_data_file : str
                                  Path to the file with point support data. Reads all files processed by the GeoPandas.

        polygon_file : str
                       Path to the file with polygon data. Reads all files processed by GeoPandas.

        point_support_geometry_col : str
                                     The name of the point support geometry column.

        point_support_val_col : str
                                The name of the point support column with values.

        polygon_geometry_col : str
                               The name of the polygon geometry column.

        polygon_index_col : str
                            The name of polygon's index column (must be unique!).

        use_point_support_crs : bool, default = True
                                If set to False then the point support crs is transformed to the same crs as polygon
                                dataset.

        dropna : bool, default = True
                 Drop areas with NaN records (id or geometry).

        point_support_layer_name : str, default = None
                                   If provided file is .gpkg then this parameter must be provided.

        polygon_layer_name  : str, default = None
                              If provided file is .gpkg then this parameter must be provided.
        """
        # Load data
        if point_support_data_file.lower().endswith('.gpkg'):
            point_support = gpd.read_file(point_support_data_file, layer=point_support_layer_name)
        else:
            point_support = gpd.read_file(point_support_data_file)
        if polygon_file.lower().endswith('.gpkg'):
            polygon_data = gpd.read_file(polygon_file, layer=polygon_layer_name)
        else:
            polygon_data = gpd.read_file(polygon_file)

        self.pointset = self.from_geodataframes(point_support,
                                                polygon_data,
                                                point_support_val_col=point_support_val_col,
                                                polygon_geometry_col=polygon_geometry_col,
                                                polygon_index_col=polygon_index_col,
                                                point_support_geometry_col=point_support_geometry_col,
                                                use_point_support_crs=use_point_support_crs,
                                                dropna=dropna)

        return self.pointset

    def from_geodataframes(self,
                           point_support_dataframe: Union[gpd.GeoDataFrame, gpd.GeoSeries],
                           polygon_dataframe: gpd.GeoDataFrame,
                           point_support_geometry_col: str,
                           point_support_val_col: str,
                           polygon_geometry_col: str,
                           polygon_index_col: str,
                           use_point_support_crs: bool = True,
                           dropna: bool = True):
        """
        Methods prepares the point support data from files.

        Parameters
        ----------
        point_support_dataframe : GeoDataFrame or GeoSeries


        polygon_dataframe : GeoDataFrame
                       Path to the file with polygon data. Reads all files processed by GeoPandas.

        point_support_geometry_col : str
                                     The name of the point support geometry column.

        point_support_val_col : str
                                The name of the point support column with values.

        point_support_val_col : str
                                The name of the point support column with values.

        polygon_geometry_col : str
                               The name of the polygon geometry column.

        polygon_index_col : str
                            The name of polygon's index column (must be unique!).

        use_point_support_crs : bool, default = True
                                If set to False then the point support crs is transformed to the same crs as polygon
                                dataset.

        dropna : bool, default = True
                 Drop areas with NaN records (id or geometry).
        """
        # Select data
        point_support, polygon_data = self._select_data(point_support_dataframe,
                                                        polygon_dataframe,
                                                        point_support_geometry_col,
                                                        point_support_val_col,
                                                        polygon_geometry_col,
                                                        polygon_index_col)

        # Transform CRS
        point_support, polygon_data = self._transform_crs(point_support, polygon_data, use_point_support_crs)

        # Merge data
        crs_ps = point_support.crs
        joined = gpd.sjoin(point_support, polygon_data, how='left', op='within')

        if dropna:
            joined.dropna(inplace=True)

        # Create output dict
        output_dict = self._joined_to_dict(joined, point_support_geometry_col, point_support_val_col)
        output_dict['info'] = {'crs': crs_ps}

        self.pointset = output_dict

        return self.pointset

    @staticmethod
    def _joined_to_dict(joined: gpd.GeoDataFrame,
                        point_geometry_col_name: str,
                        point_value_col_name: str,
                        polygon_index_name: str = 'index_right') -> Dict:
        """
        Method transforms merged GeoDataFrames into a dictionary.

        Parameters
        ----------
        joined : gpd.GeoDataFrame
                 Merged Points and Polygons.

        point_geometry_col_name : str
                                  The name of the column with geometry (Point).

        point_value_col_name : str
                               The name of the column with values (from the point support).

        polygon_index_name : str, default="index_right"
                             Column with the polygon data indexes.

        Returns
        -------
        output_d : Dict
        """
        # TODO: Fn skips points that are not assigned to any area, maybe log it somewhere...

        indexes = pd.unique(joined[polygon_index_name])

        output_d = dict()
        output_d['data'] = {}

        for idx in indexes:
            points_within_area = joined[joined[polygon_index_name] == idx]
            points_within_area['x'] = points_within_area[point_geometry_col_name].x
            points_within_area['y'] = points_within_area[point_geometry_col_name].y
            points = points_within_area[['x', 'y', point_value_col_name]]
            points = points.to_numpy()
            output_d['data'][idx] = points

        return output_d

    def _select_data(self,
                     point_support: gpd.GeoDataFrame,
                     polygon_data: gpd.GeoDataFrame,
                     point_support_geometry_col: str,
                     point_support_value_col: str,
                     polygon_data_geometry_col: str,
                     polygon_data_index_col: str):
        point_support = self._select_point_support_data(point_support,
                                                        point_support_geometry_col,
                                                        point_support_value_col)
        polygon_data = self._select_polygon_data(polygon_data,
                                                 polygon_data_geometry_col,
                                                 polygon_data_index_col)
        return point_support, polygon_data

    @staticmethod
    def _select_point_support_data(point_support: gpd.GeoDataFrame,
                                   geometry_col: str,
                                   value_col: str):
        point_support = point_support[[geometry_col, value_col]]
        return point_support

    @staticmethod
    def _select_polygon_data(polygon_data: gpd.GeoDataFrame,
                             geometry_col: str,
                             index_col: str):
        polygon_data = polygon_data[[geometry_col, index_col]]
        polygon_data.set_index(index_col, inplace=True)
        return polygon_data

    @staticmethod
    def _transform_crs(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, use_points: bool):
        if use_points:
            if polygons.crs != points.crs:
                polygons = polygons.to_crs(points.crs)
        else:
            if polygons.crs != points.crs:
                points = points.to_crs(polygons.crs)
        return points, polygons


def get_point_support_from_files(point_support_data_file: str,
                                 polygon_file: str,
                                 point_support_geometry_col: str,
                                 point_support_val_col: str,
                                 polygon_geometry_col: str,
                                 polygon_index_col: str,
                                 use_point_support_crs: bool = True,
                                 dropna: bool = True,
                                 point_support_layer_name: str = None,
                                 polygon_layer_name: str = None) -> Dict:
    """
    Function prepares the point support data from files.

    Parameters
    ----------
    point_support_data_file : str
                              Path to the file with point support data. Reads all files processed by the GeoPandas.

    polygon_file : str
                   Path to the file with polygon data. Reads all files processed by GeoPandas.

    point_support_geometry_col : str
                                 The name of the point support geometry column.

    point_support_val_col : str
                            The name of the point support column with values.

    polygon_geometry_col : str
                           The name of the polygon geometry column.

    polygon_index_col : str
                        The name of polygon's index column (must be unique!).

    use_point_support_crs : bool, default = True
                            If set to False then the point support crs is transformed to the same crs as polygon
                            dataset.

    dropna : bool, default = True
             Drop areas with NaN records (id or geometry).

    point_support_layer_name : str, default = None
                               If provided file is .gpkg then this parameter must be provided.

    polygon_layer_name  : str, default = None
                          If provided file is .gpkg then this parameter must be provided.

    Returns
    -------
    pointset : Dict
               Prepared dict with Polygon data. It's structure is:
               polyset = {
                   'data': {id: [[point support array - lon, lat, value]], },
                   'info': {
                       'crs': CRS of a dataset
                   }
               }
    """
    point_support = PointSupportDataClass()
    pointset = point_support.from_files(point_support_data_file, polygon_file,
                                        point_support_geometry_col=point_support_geometry_col,
                                        point_support_val_col=point_support_val_col,
                                        polygon_geometry_col=polygon_geometry_col,
                                        polygon_index_col=polygon_index_col,
                                        use_point_support_crs=use_point_support_crs,
                                        dropna=dropna,
                                        point_support_layer_name=point_support_layer_name,
                                        polygon_layer_name=polygon_layer_name)
    return pointset


def get_point_support_from_geodataframes(point_support_dataframe: Union[gpd.GeoDataFrame, gpd.GeoSeries],
                                         polygon_dataframe: gpd.GeoDataFrame,
                                         point_support_geometry_col: str,
                                         point_support_val_col: str,
                                         polygon_geometry_col: str,
                                         polygon_index_col: str,
                                         use_point_support_crs: bool = True,
                                         dropna: bool = True) -> Dict:
    """
    Function prepares the point support data from files.

    Parameters
    ----------
    point_support_dataframe : GeoDataFrame or GeoSeries


    polygon_dataframe : GeoDataFrame
                   Path to the file with polygon data. Reads all files processed by GeoPandas.

    point_support_geometry_col : str
                                 The name of the point support geometry column.

    point_support_val_col : str
                            The name of the point support column with values.

    point_support_val_col : str
                            The name of the point support column with values.

    polygon_geometry_col : str
                           The name of the polygon geometry column.

    polygon_index_col : str
                        The name of polygon's index column (must be unique!).

    use_point_support_crs : bool, default = True
                            If set to False then the point support crs is transformed to the same crs as polygon
                            dataset.

    dropna : bool, default = True
             Drop areas with NaN records (id or geometry).

    Returns
    -------
    pointset : Dict
               Prepared dict with Polygon data. It's structure is:
               polyset = {
                   'data': {id: [[point support array - lon, lat, value]], },
                   'info': {
                       'crs': CRS of a dataset
                   }
               }
    """
    point_support = PointSupportDataClass()
    pointset = point_support.from_geodataframes(point_support_dataframe,
                                                polygon_dataframe,
                                                point_support_geometry_col=point_support_geometry_col,
                                                point_support_val_col=point_support_val_col,
                                                polygon_geometry_col=polygon_geometry_col,
                                                polygon_index_col=polygon_index_col,
                                                use_point_support_crs=use_point_support_crs,
                                                dropna=dropna)
    return pointset
