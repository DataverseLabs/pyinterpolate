from typing import Union

import geopandas as gpd
import numpy as np

from pyinterpolate.processing.utils.exceptions import IndexColNotUniqueError, WrongGeometryTypeError


class PointSupportDataClass:
    """
    Class stores and describes the point support data in relation to polygon dataset.

    Attributes
    ----------
    pointset : dict
               Prepared dict with Polygon data. It's structure is:
               polyset = {
                   'area id': [point support array - lon, lat, value]
                   'info': {
                       'crs': CRS of a dataset
                   }
               }

    Methods
    -------
    from_files()
        Loads point support and polygon data from files.

    from_dataframes()
        Loads point support and polygon data from dataframe.

    group_point_support()
        Groups point support based on the given polygon data.

    load_point_support()
        Loads point support data only.

    Examples
    --------

    """
    def __init__(self):
        self.pointset = None
        self.point_support = None
        self.polygon_data = None

    def from_files(self,
                   point_support_data_file: str,
                   polygon_file: str,
                   point_support_val_col: str,
                   polygon_geometry_col: str,
                   polygon_index_col: str,
                   point_support_geometry_col: str,
                   use_point_support_crs: bool = True):
        """
        Methods prepares the point support data from files.

        Parameters
        ----------
        point_support_data_file : str
                                  Path to the file with point support data. Reads all files processed by the GeoPandas.

        polygon_file : str
                       Path to the file with polygon data. Reads all files processed by GeoPandas.

        point_support_val_col : str
                                The name of the point support column with values.

        polygon_geometry_col : str
                               The name of the polygon geometry column.

        polygon_index_col : str
                            The name of polygon's index column (must be unique!).

        point_support_geometry_col : str
                                     The name of the point support geometry column.

        use_point_support_crs : bool, default = True
                                If set to False then the point support crs is transformed to the same crs as polygon
                                dataset.
        """
        pass

    def from_geodataframes(self,
                        point_support_dataframe: Union[gpd.GeoDataFrame, gpd.GeoSeries],
                        polygon_dataframe: gpd.GeoDataFrame,
                        point_support_val_col: str,
                        polygon_geometry_col: str,
                        polygon_index_col: str,
                        point_support_geometry_col: str,
                        use_point_support_crs: bool = True):
        """
        Methods prepares the point support data from files.

        Parameters
        ----------
        point_support_dataframe : GeoDataFrame or GeoSeries


        polygon_dataframe : GeoDataFrame
                       Path to the file with polygon data. Reads all files processed by GeoPandas.

        point_support_val_col : str
                                The name of the point support column with values.

        polygon_geometry_col : str
                               The name of the polygon geometry column.

        polygon_index_col : str
                            The name of polygon's index column (must be unique!).

        point_support_geometry_col : str
                                     The name of the point support geometry column.

        use_point_support_crs : bool, default = True
                                If set to False then the point support crs is transformed to the same crs as polygon
                                dataset.
        """