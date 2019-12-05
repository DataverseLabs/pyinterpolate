import numpy as np
from geopandas.tools import sjoin
import pyproj

from pyinterpolate.kriging.helper_functions.get_centroids import get_centroids


class PKData:
    """
    Class to prepare data for Poisson Kriging.

    INPUT PARAMETERS:
    :param counties_data: geodataframe object created with geopandas which contains polygons and respective rates,
    :param population data: geodataframe objected created with geopandas which contains points and population values,
    :param areal_id_col_name: string with the id column name from counties_data geodataframe,
    :param areal_val_name: string with the rate column name from counties_data geodataframe,
    :param areal_nan_val: value of the NaN value inside the dataset,
    :param population_val_col_name: string with the population counts column name from population_data geodataframe.

    ATTRIBUTES:
        initial_dataset: geodataframe from counties_data parameter,
        counties_dataset: geodataframe derived from the initial_dataset without rows with NaN values,
        population_dataset: geodataframe from population_data parameter,
        joined_datasets: geodataframe with joined counties (areal) and population data, points not assigned to
            any counties are dropped,
        id_col: areal id column name from areal_id_col_name parameter,
        val_col: areal rate column name from areal_val_name parameter,
        pop_col: population values column name from population_val_column_name parameter,
        total_population_per_unit: pandas Series with index derived from the counties_dataset geodataframe and
            sums of population per area,
        centroids_of_areal_data: numpy array of values [point x, point y, value, area id].

    PRIVATE METHODS:
        _join_datasets(self): method performs sjoin on yhe areal and population datasets (it groups points of
            population into respective areas' polygons).
        _get_tot_population(self, area_id_col, population_val_col): metod calculates total population per area.

    STATIC METHODS:
        get_areal_centroids(counties, vals, ids): method gets centroids from the respective polygons and returns
            numpy array with [centroid position x, centroid position y, rate, area id].
    """

    def __init__(self, counties_data, population_data,
                 areal_id_col_name, areal_val_name,
                 areal_nan_val,
                 population_val_col_name):
        self.initial_dataset = counties_data

        # Remove rows with nan value from the dataset
        self.counties_dataset = self.initial_dataset[
            self.initial_dataset[areal_val_name] != areal_nan_val]

        self.population_dataset = population_data
        self.joined_datasets = self._join_datasets()

        # Clear joined dataset from nans
        self.joined_datasets = self.joined_datasets[
            self.joined_datasets['index_right'].notna()]

        self.id_col = areal_id_col_name
        self.val_col = areal_val_name
        self.pop_col = population_val_col_name
        self.total_population_per_unit = self._get_tot_population(self.id_col,
                                                                  self.pop_col)
        self.centroids_of_areal_data = self.get_areal_centroids(self.counties_dataset,
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
    def get_areal_centroids(counties, vals, ids):
        """Function get centroids and remove nan values from the dataset"""

        c = get_centroids(counties, vals, ids)
        c = c[~np.isnan(c).any(axis=1)]
        return c
