import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyinterpolate.kriging.areal_poisson_kriging.area_to_area.ata_poisson_kriging import AtAPoissonKriging
from pyinterpolate.kriging.areal_poisson_kriging.area_to_point.atp_poisson_kriging import AtPPoissonKriging


class ArealKriging:

    def __init__(self, semivariogram_model, known_areas, known_areas_points, kriging_type='ata'):
        """
        :param semivariogram_model: (Theoretical Semivariogram) Theoretical Semivariogram used for data interpolation,
        :param known_areas: (numpy array) array of areas in the form:
            [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value]
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
        else:
            self.k_func = kriging_types[self.ktype](self.semivar_model,
                                                    self.areal_data_known,
                                                    self.point_counts_within_area)
            self.reg_model = kriging_types[self.ktype]

        self.regularized = None  # Regularized

    def predict(self, unknown_area_points, number_of_neighbours, max_search_radius):
        """
        Function predicts areal value in a unknown location based on the area-to-area or area-to-point Poisson Kriging
        :param unknown_area_points: (numpy array) array of points within an unknown area in the form:
            [area_id, [point_position_x, point_position_y, value]]
        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to number of
            neighbors).
        :return: prediction, error, estimated mean, weights:
            [value in unknown location, error, estimated mean, weights]
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

    def regularize_data(self, number_of_neighbours, s_radius, data_crs="EPSG:4326"):
        """
        Function regularizes whole dataset and creates new values and error maps based on the kriging type.
        If chosen type is area to area then function returns geopandas dataframe with area id, areal geometry,
        estimated value, estimated prediction error, RMSE of prediction.
        If chosen type is area to point then function returns geopandas dataframe with area id, point coordinates,
        estimated value, estimated prediction error, RMSE of areal prediction.

        Function do not predict unknown values, areas with NaN's are skipped.

        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param s_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to number of
            neighbors),
        :param data_crs: (string) data crs, look into: https://geopandas.org/projections.html
        :return map_dataframe: (GeoPandas GeoDataFrame object)
        """
        areas_ids = self.areal_data_known[:, 0]

        list_of_vals = []
        for a_id in areas_ids:
            prediction_rows = self._get_prediction_row(a_id, number_of_neighbours, s_radius)

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
        gdf.columns = ['id', 'geometry', 'estimated value', 'estimated prediction error', 'rmse']
        gdf.crs = data_crs

        return gdf


if __name__ == '__main__':
    pass
    # from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import
    # calculate_weighted_semivariance
    # from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
    #
    # from sample_data.data import Data
    # from pyinterpolate.data_processing.data_preparation.prepare_areal_shapefile import prepare_areal_shapefile
    # from pyinterpolate.data_processing import get_points_within_area
    # from pyinterpolate.data_processing import set_areal_weights
    #
    # # DATA PREPARATION
    #
    # areal_dataset = Data().poland_areas_dataset
    # subset = Data().poland_population_dataset
    #
    # a_id = 'IDx'
    # areal_val = 'LB RATES 2'
    # points_val = 'TOT'
    #
    # maximum_range = 100000
    # step_size = 8000
    # lags = np.arange(0, maximum_range, step_size)
    #
    # areal_data_prepared = prepare_areal_shapefile(areal_dataset, a_id, areal_val)
    # points_in_area = get_points_within_area(areal_dataset, subset, areal_id_col_name=a_id,
    #                                         points_val_col_name=points_val)
    #
    # # Semivariance deconvolution
    #
    # semivar_modeling_data = set_areal_weights(areal_data_prepared, points_in_area)
    # smv_model = calculate_weighted_semivariance(semivar_modeling_data, lags, step_size)
    #
    # semivariogram = TheoreticalSemivariogram(areal_data_prepared[:, 2:], smv_model)
    #
    # semivariogram.find_optimal_model()
    #
    # # Poisson Kriging
    #
    # search_radius = 30000
    # number_of_observations = 6
    #
    # # TEST REGULARIZATION
    #
    # # GET DATA CRS
    #
    # data_file = gpd.read_file(areal_dataset)
    # crs = data_file.crs
    #
    # # ATA
    # ak = ArealKriging(semivariogram, areal_data_prepared, points_in_area, kriging_type='atp')
    # dataframe = ak.regularize_data(number_of_neighbours=number_of_observations, s_radius=search_radius, data_crs=crs)
    #
    # print(dataframe.head())

    # Get one area as unknown
    # unknown_area_id = [101]
    #
    # u_area = areal_data_prepared[areal_data_prepared[:, 0] == unknown_area_id][0]
    # u_points = points_in_area[points_in_area[:, 0] == unknown_area_id][0]
    #
    # k_areas = areal_data_prepared[areal_data_prepared[:, 0] != unknown_area_id]
    # k_points = points_in_area[points_in_area[:, 0] != unknown_area_id]
    #
    # # Semivariance deconvolution
    #
    # semivar_modeling_data = set_areal_weights(k_areas, k_points)
    # smv_model = calculate_weighted_semivariance(semivar_modeling_data, lags, step_size)
    #
    # semivariogram = TheoreticalSemivariogram(k_areas[:, 2:], smv_model)
    #
    # semivariogram.find_optimal_model()
