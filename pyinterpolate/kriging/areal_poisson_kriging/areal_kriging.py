import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from pyinterpolate.kriging.areal_poisson_kriging.area_to_area.ata_poisson_kriging import AtAPoissonKriging
from pyinterpolate.kriging.areal_poisson_kriging.area_to_point.atp_poisson_kriging import AtPPoissonKriging
from pyinterpolate.transform.tests import does_variogram_exist


class ArealKriging:
    """
    Class performs kriging of areas with point support data within those polygons.

    INITIALIZATION PARAMS:

    :param semivariogram_model: (TheoreticalSemivariogram) Theoretical Semivariogram used for data interpolation,
    :param known_areas: (numpy array) areas in the form:
        [area_id, polygon, centroid x, centroid y, value]
    :param known_areas_points: (numpy array) points within areas in the form:
        [area_id, [point_position_x, point_position_y, value]],
    :param kriging_type: (str) default 'ata'; 'ata' - area to area poisson kriging,
        'atp' - area to point poisson kriging.
    """

    def __init__(self, semivariogram_model, known_areas, known_areas_points, kriging_type='ata'):
        """
        :param semivariogram_model: (Theoretical Semivariogram) Theoretical Semivariogram used for data interpolation,
        :param known_areas: (numpy array) areas in the form:
            [area_id, polygon, centroid x, centroid y, value]
        :param known_areas_points: (numpy array) array of points within areas in the form:
            [area_id, [point_position_x, point_position_y, value]],
        :param kriging_type: (string) default 'ata'.
            'ata' - area to area poisson kriging,
            'atp' - area to point poisson kriging.
        """

        kriging_types = {
            'ata': AtAPoissonKriging,
            'atp': AtPPoissonKriging
        }

        # Check if semivariogram exists
        does_variogram_exist(semivariogram_model)
        self.semivar_model = semivariogram_model
        self.areal_data_known = known_areas
        self.point_counts_within_area = known_areas_points
        self.ktype = kriging_type

        # Check kriging type

        if kriging_type not in kriging_types.keys():
            l1 = 'Provided argument is not correct. You must choose kriging type.\n'
            l2 = "'ata' - area to area poisson kriging,\n"
            l3 = "'atp' - area to point poisson kriging."
            message = l1 + l2 + l3
            raise TypeError(message)

        self.k_func = kriging_types[self.ktype](self.semivar_model,
                                                self.areal_data_known,
                                                self.point_counts_within_area)
        self.reg_model = kriging_types[self.ktype]

        self.regularized = None  # Regularized

    def predict(self, unknown_area_points, number_of_neighbours, max_search_radius):
        """
        Function predicts area value in an unknown location based on the area-to-area or area-to-point Poisson Kriging.

        INPUT:

        :param unknown_area_points: (numpy array) points within an unknown area in the form:
            [area_id, [point_position_x, point_position_y, value]],
        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to
            the number_of_neighbors).

        OUTPUT:

        :return: prediction, error, estimated mean, weights:
            [value_in_unknown_location, error, estimated_mean, weights].
        """

        prediction = self.k_func.predict(unknown_area_points,
                                         number_of_neighbours,
                                         max_search_radius)

        return prediction

    @staticmethod
    def _rmse_areal(prediction_row, u_area_val):

        est_val = prediction_row[0]
        rmse = np.sqrt((u_area_val - est_val)**2)

        prediction_vals = [
            prediction_row[0],
            prediction_row[1],
            rmse
        ]  # Estimated value, Estimated error, RMSE

        return np.array(prediction_vals)

    @staticmethod
    def _rmse_area_to_point(prediction_row, u_areal_val):
        estimated_areal_sum = np.sum(prediction_row[:, 0])
        rmse = np.sqrt((u_areal_val - estimated_areal_sum)**2)
        rmses = rmse * np.ones(prediction_row.shape[0])
        prediction_vals = np.c_[
            prediction_row[:, -1],
            prediction_row[:, :2],
            rmses
        ]  # points xy, Estimated value, Estimated error, RMSEs

        return prediction_vals

    def _get_prediction_row(self, regularized_area_id, n_neighbours, radius):

        # Divide dataset for prediction
        u_area = self.areal_data_known[self.areal_data_known[:, 0] == regularized_area_id][0]
        u_points = self.point_counts_within_area[self.point_counts_within_area[:, 0] == regularized_area_id][0]

        k_areas = self.areal_data_known[self.areal_data_known[:, 0] != regularized_area_id]
        k_points = self.point_counts_within_area[self.point_counts_within_area[:, 0] != regularized_area_id]

        # Prepare model

        temp_model = self.reg_model(
            self.semivar_model,
            k_areas,
            k_points
        )

        # Predict

        prediction = temp_model.predict(u_points,
                                        n_neighbours,
                                        radius)

        # Add RMSE of prediction

        if self.ktype == 'ata':
            output_rows = self._rmse_areal(prediction, u_area[-1])
        else:
            output_rows = self._rmse_area_to_point(prediction, u_area[-1])

        return output_rows

    def regularize_data(self, number_of_neighbours, max_search_radius, data_crs="EPSG:4326"):
        """
        Function regularizes whole dataset and creates new values and error maps based on the kriging type.
            If chosen type is area to area then function returns Geopandas GeoDataFrame with area id, areal geometry,
            estimated value, estimated prediction error, RMSE of prediction.
            If chosen type is area to point then function returns Geopandas GeoDataFrame with area id, point
                coordinates, estimated value, estimated prediction error, RMSE of areal prediction.

        Function do not predict unknown values, areas with NaN are skipped.

        INPUT:

        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to
            number_of_neighbours),
        :param data_crs: (str) data crs, look into: https://geopandas.org/projections.html

        OUTPUT:

        :return: regularized dataset (GeoPandas GeoDataFrame object)
        """
        areas_ids = self.areal_data_known[:, 0]

        list_of_vals = []
        for a_id in areas_ids:
            prediction_rows = self._get_prediction_row(a_id, number_of_neighbours, max_search_radius)

            # Add id and geometry into a list

            if self.ktype == 'ata':
                geometry = self.areal_data_known[self.areal_data_known[:, 0] == a_id]
                geometry = geometry[0][1]
                data_row = [a_id, geometry, prediction_rows[0], prediction_rows[1], prediction_rows[2]]
                list_of_vals.append(data_row)
            else:
                for val in prediction_rows:
                    xy = Point(val[0])
                    list_of_vals.append([a_id, xy, val[1], val[2], val[3]])

        # Transform array into a dataframe

        gdf = gpd.GeoDataFrame(list_of_vals)
        gdf.columns = ['id', 'geometry', 'reg.est', 'reg.err', 'rmse']
        gdf.crs = data_crs

        return gdf
