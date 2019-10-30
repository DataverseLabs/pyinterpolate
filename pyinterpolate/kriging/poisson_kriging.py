import geopandas as gpd
from geopandas.tools import sjoin
import pyproj


class PKrige:
    """
    Class for Poisson Kriging, Area-to-area (ATA) and Area-to-Point (ATP) Poisson Kriging interpolation of
    the unknown values in a given location (position). Class takes two arguments during the initialization:
    counties_data -
    population_data -
    semivariogram_model -

    Available methods:


    Static methods:


    Data visualization methods:


    Example how to prepare model:



    """

    def __init__(self, counties_data, population_data, semivariogram_model,
                 areal_id_col_name, population_val_col_name):
        """
        :param counties_data:
        :param population_data:
        """
        self.semivariogram = semivariogram_model
        self.counties_dataset = gpd.read_file(counties_data)
        self.population_dataset = gpd.read_file(population_data)
        self.joined_datasets = self._join_datasets()
        self.total_population_per_unit = self._get_tot_population(areal_id_col_name,
                                                                  population_val_col_name)

    # Data processing private class methods

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
