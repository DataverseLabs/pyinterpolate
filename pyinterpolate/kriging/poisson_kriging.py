import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
import pyproj

from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance
from pyinterpolate.kriging.helper_functions.get_centroids import get_centroids


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
                 areal_id_col_name, areal_val_name, population_val_col_name):
        """
        :param counties_data: address to dataset with areal data (polygons),
        :param population_data: address to dataset with point data (centroids),
        :param semivariogram_model: deconvoluted semivariogram,
        :param areal_id_col_name: id column name of areal data,
        :param areal_val_name: value column name of areal data,
        :param population_val_col_name: value column name of population data,

        """
        self.semivariogram = semivariogram_model
        self.counties_dataset = gpd.read_file(counties_data)
        self.population_dataset = gpd.read_file(population_data)
        self.joined_datasets = self._join_datasets()
        self.id_col = areal_id_col_name
        self.val_col = areal_val_name
        self.pop_col = population_val_col_name
        self.total_population_per_unit = self._get_tot_population(self.id_col,
                                                                  self.pop_col)
        self.centroids_of_areal_data = self._get_areal_centroids(self.counties_dataset,
                                                                 self.val_col,
                                                                 self.id_col)
        self.prepared_data = None
        self.global_mean = self.counties_dataset[self.val_col].mean()  # Only for test purposes

        # Parameters
        self.lags = None
        self.step = None
        self.max_no_of_observations = None
        self.max_search_radius = None

    # Data preparation functions

    def set_params(self,
                   lags_number,
                   lag_step_size,
                   max_no_of_observations,
                   search_radius
                   ):
        # Function sets class parameters

        self.lags = lags_number
        self.step = lag_step_size
        self.max_no_of_observations = max_no_of_observations
        self.max_search_radius = search_radius
        print('Parameters have been set')

    def prepare_data_areal(self, unknown_area_id, verbose=False):
        """
        :param unknown_area_id: id of the area with unknown value,
        :param verbose: if True then method informs about the successful operation.
        :return output_data: prepared dataset which contains:
        [[known_position_x, known_position_y, value, area id, distance_to_unknown_position], [...]]
        """

        # Simple Poisson Kriging (with centroid-based approach)

        # Get unknown area centroid

        centroid_pos = self.centroids_of_areal_data[
            self.centroids_of_areal_data[:, -1] == unknown_area_id]
        cx_cy = centroid_pos[0][:2]

        # Get unknown area population

        unknown_population = self.total_population_per_unit[unknown_area_id]

        # Calculate distances

        r = np.array([cx_cy])
        known_centroids = self.centroids_of_areal_data[
            self.centroids_of_areal_data[:, -1] != unknown_area_id]
        kc = known_centroids[:, :2]

        distances_array = np.zeros(kc.shape)
        for i in range(0, r.shape[1]):
            distances_array[:, i] = (kc[:, i] - r[:, i]) ** 2
        s = distances_array.sum(axis=1)
        s = np.sqrt(s)
        s = s.T

        # Build set for Poisson Kriging

        kriging_data = np.c_[known_centroids, s]  # [coo_x, coo_y, val, dist_to_unkn]

        # remove nans
        kriging_data = kriging_data[~np.isnan(kriging_data).any(axis=1)]

        # sort by distance
        kriging_data = kriging_data[kriging_data[:, -1].argsort()]

        # set output by number of neighbours
        # TODO: account for a distance

        output_data = kriging_data[:self.max_no_of_observations]
        self.prepared_data = np.array(output_data)

        if verbose:
            print(('Centroid of area id {} prepared for processing').format(
                unknown_area_id))

        return output_data

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

    def _get_areal_centroids(self, counties, vals, ids):
        """Function get centroids and remove nan values from the dataset"""

        c = get_centroids(counties, vals, ids)
        c = c[~np.isnan(c).any(axis=1)]
        return c

    def _get_neighbours_ids(self):
        pass

    def _calculate_weighted_mean(self):
        pass

    # Modeling functions

    def poisson_kriging(self):
        """
        To run kriging operation prepare_data method should be invoked first
        :return zhat, sigma, w[-1][0], w:
        [value in unknown location, error, estimated mean, weights, area_id]
        """
        n = len(self.prepared_data)
        k = np.array([self.prepared_data[:, -1]])
        k = k.T
        k1 = np.matrix(1)
        k = np.concatenate((k, k1), axis=0)

        distances = calculate_distance(self.prepared_data[:, :2])
        predicted = self.model.predict(distances.ravel())
        print(predicted)

    def area_to_area_pk(self):
        pass

    def area_to_point_pk(self):
        pass

    # Temporary functions

    def _set_fake_weight_array(self, weights):
        # Function used to create temp rates means - test purposes only
        arr = np.diag(weights)
        return arr