import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
import pyproj

from pyinterpolate.kriging.helper_functions.get_centroids import get_centroids


class PKData:
    """
    Class instance represents data prepared for Poisson Kriging
    """

    def __init__(self, counties_data, population_data,
                 areal_id_col_name, areal_val_name,
                 areal_nan_val,
                 population_val_col_name):
        self.initial_dataset = gpd.read_file(counties_data)

        # Remove rows with nan value from the dataset
        self.counties_dataset = self.initial_dataset[
            self.initial_dataset[areal_val_name] != areal_nan_val]

        self.population_dataset = gpd.read_file(population_data)
        self.joined_datasets = self._join_datasets()

        # Clear joined dataset from nans
        self.joined_datasets = self.joined_datasets[
            self.joined_datasets['index_right'].notna()]

        self.id_col = areal_id_col_name
        self.val_col = areal_val_name
        self.pop_col = population_val_col_name
        self.total_population_per_unit = self._get_tot_population(self.id_col,
                                                                  self.pop_col)
        self.centroids_of_areal_data = self._get_areal_centroids(self.counties_dataset,
                                                                 self.val_col,
                                                                 self.id_col)

    def _join_datasets(self):
        """Function perform left join of two spatial datasets. Method is useful when someone is interested in
        the relation between point spatial data (population centroids) and polygons containing those points
        (counties).

        Both geodataframes must have the same coordinate reference system (crs). Test is performed before join.
        If crs is not the same then population centroids are transformed.

        OUTPUT:
        :return df: geodataframe of points with column indicating index from the counties geodataframe.
        """

        # Check crs
        if not pyproj.Proj(self.counties_dataset.crs).is_exact_same(pyproj.Proj(self.population_dataset.crs)):
            population = self.population_dataset.to_crs(self.counties_dataset.crs)
            df = sjoin(population, self.counties_dataset, how='left')
        else:
            df = sjoin(self.population_dataset, self.counties_dataset, how='left')
        return df

    def _get_tot_population(self, area_id_col, population_val_col):
        """Function calculate total population per administrative unit and returns dataframe with
        area index | sum of population

        INPUT:
        :param area_id_col: name of the id column from counties dataset,
        :param population_val_col: name of the column with population values in the population dataset.

        OUTPUT:
        :return tot_pop_series: series with total population per area.
        """
        tot_pop = self.joined_datasets.groupby([area_id_col]).sum()
        tot_pop_series = tot_pop[population_val_col]
        return tot_pop_series

    @staticmethod
    def _get_areal_centroids(counties, vals, ids):
        """Function get centroids and remove nan values from the dataset"""

        c = get_centroids(counties, vals, ids)
        c = c[~np.isnan(c).any(axis=1)]
        return c
