import numpy as np

from pyinterpolate.transform.prepare_kriging_data import prepare_ata_known_areas
from pyinterpolate.transform.prepare_kriging_data import prepare_atp_data


class WeightedBlock2BlockSemivariance:

    def __init__(self, semivariance_model):
        self.semivariance_model = semivariance_model

    def _avg_smv(self, datarows):
        """
        Function calculates weight and partial semivariance of a data row [point value a, point value b, distance]
        :param datarows: (numpy arrays) numpy arrays of [point value a, point value b, distance between points]
        :return: (float, float) weighted semivariance and single weight
        """
        weighted_semivariance_sum = 0
        sum_of_weights = 0

        for datarow in datarows:
            single_weight = datarow[0] * datarow[1]
            partial_semivar = self.semivariance_model.predict(datarow[-1])
            weighted_semivar = partial_semivar * single_weight
            weighted_semivariance_sum = weighted_semivariance_sum + weighted_semivar
            sum_of_weights = sum_of_weights + single_weight

        return weighted_semivariance_sum, sum_of_weights

    def calculate_average_semivariance(self, data_points):
        """
        Function calculates average semivariance. Calculation is performed based on the equation:

            (1 / SUM(Pi-s=1) SUM(Pj-s'=1) w_(ss')) * SUM(Pi-s=1) SUM(Pj-s'=1) w_(ss') * semivariance(distance)

        :return: (numpy array) array of predicted and weighted semivariances
        """
        k = []
        for prediction_input in data_points:
            w_sem_sum = 0
            w_sem_weights_sum = 0
            for single_area in prediction_input:
                w_sem = self._avg_smv(single_area)
                w_sem_sum = w_sem_sum + w_sem[0]
                w_sem_weights_sum = w_sem_weights_sum + w_sem[1]
            k.append(w_sem_sum / w_sem_weights_sum)
        return np.array(k)


class WeightedBlock2PointSemivariance:

    def __init__(self, semivariance_model):
        self.semivariance_model = semivariance_model

    def _avg_smv(self, unknown_pt_value, known_points_values_and_distances):
        """
        Function calculates weight and partial semivariance of a data row [point value a, point value b, distance]
        :param unknown_pt_value: (float) value of an unknown point,
        :param known_points_values_and_distances: (numpy array) numpy array of known points values and their distances
                                                                to the unknown point,
        :return: (float) weighted semivariance for a given point
        """

        if unknown_pt_value > 0:
            partial_semivars = np.array(
                [self.semivariance_model.predict(x) for x in known_points_values_and_distances[:, 1]]
            )

            all_weights = unknown_pt_value * known_points_values_and_distances[:, 0]

            weighted_semivars = np.sum(partial_semivars * all_weights)
            weighted_block_smv = weighted_semivars / np.sum(all_weights)
        else:
            weighted_block_smv = 0

        return weighted_block_smv

    def calculate_average_semivariance(self, data_points):
        """
        Function calculates average semivariance between block (Pi) and single point from a different block Pj.
        Calculation is performed based on the equation:

            (1 / SUM(Pi-s=1) (Pj-s'=1) w_(ss')) * SUM(Pi-s=1) (Pj-s'=1) w_(ss') * semivariance(distance)

        :param data_points: (numpy array) values of points of unknown location and arrays of values of known points
                                          and distances to these points from an uknown point,
        :return: (numpy array) array of predicted and weighted semivariances for each passed point
        """

        point_to_blocks_smvs = []
        for known_area in data_points:
            pts_per_area = []
            for single_unknown_point in known_area:
                pt_output = self._avg_smv(single_unknown_point[0], single_unknown_point[1])
                pts_per_area.append(pt_output)
            point_to_blocks_smvs.append(pts_per_area)
        return np.array(point_to_blocks_smvs)


class AtPPoissonKriging:

    def __init__(self, regularized_model=None, known_areas=None, known_areas_points=None):
        """

        :param regularized_model: (RegularizedSemivariogram object) Semivariogram model after deconvolution procedure,
        :param known_areas: (numpy array) array of areas in the form:
            [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value]
        :param known_areas_points: (numpy array) array of points within areas in the form:
            [area_id, [point_position_x, point_position_y, value]]
        """

        self.model = regularized_model
        self.known_areas = known_areas
        self.known_areas = self.known_areas[self.known_areas[:, 0].argsort()]  # Sort by id
        self.known_areas_points = known_areas_points
        self.known_areas_points = self.known_areas_points[self.known_areas_points[:, 0].argsort()]  # Sort by id
        self.unknown_points = None
        self.unknown_area_total_population = None

        self.block_to_block_smv = None
        self.block_to_point_smv = None
        self.prepared_data = None

    @staticmethod
    def _add_one_to_vecs(vecs_list):
        ones = np.ones(np.shape(vecs_list)[1])
        list_with_ones = np.vstack((vecs_list, ones))
        return list_with_ones

    def predict(self, unknown_location_points,
                number_of_neighbours, max_search_radius):
        """
        Function predicts areal value in a unknown location based on the centroid-based Poisson Kriging
        :param unknown_location_points: (numpy array) array of points within an unknown area in the form:
            [area_id, [point_position_x, point_position_y, value]]
        :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
        :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
            smaller than number_of_neighbours parameter then additional neighbours are included up to number of
            neighbors).
        :return: prediction, error, estimated mean, weights:
            [value in unknown location, error, estimated mean, weights, point coordinates]
        """

        # Get total population of the unknown area
        self.unknown_area_total_population = np.sum(unknown_location_points[1][:, -1])

        self.prepared_data, self.unknown_points = prepare_atp_data(
            points_within_unknown_area=unknown_location_points,
            known_areas=self.known_areas, points_within_known_areas=self.known_areas_points,
            number_of_neighbours=number_of_neighbours, max_search_radius=max_search_radius
        )
        # [[id (known), val, [
        #                    [unknown_point_value
        #                        [known pt vals, distances between unknown point and known points]],
        #                    [...],
        #                   ],
        #  total points value], [list of unknown point coordinates]]

        self.block_to_point_smv = WeightedBlock2PointSemivariance(semivariance_model=self.model)
        self.block_to_block_smv = WeightedBlock2BlockSemivariance(semivariance_model=self.model)

        # Calculate Average Point to Area semivariance for each area - len of dataset
        u_pts_and_areas = self.prepared_data[:, 2]
        k = self.block_to_point_smv.calculate_average_semivariance(u_pts_and_areas)

        # ADD ONE TO THE EACH VECTOR

        nk = self._add_one_to_vecs(k)

        # Calculate block to block average semivariance between known areas

        # Prepare areas blocks for calculation

        block_ids = self.prepared_data[:, 0]

        points_of_interest = []
        for area_id in block_ids:
            points_of_interest.append(self.known_areas_points[self.known_areas_points[:, 0] == area_id])

        # Calculate distances between blocks / prepare data for prediction
        # [id base,
        #     [
        #         id other,
        #         [base point value, other point value, distance between points]
        #     ]
        #  ]

        distances_of_chosen_areas = prepare_ata_known_areas(points_of_interest)

        predicted = self._calculate_avg_semivars_between_known(distances_of_chosen_areas)

        # Add weights to predicted values (diagonal)

        weights_mtx = self.calculate_weight_arr()
        predicted = predicted + weights_mtx

        # Prepare weights matrix

        p_ones = np.ones((predicted.shape[0], 1))
        predicted_with_ones_col = np.c_[predicted, p_ones]
        p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
        p_ones_row[0][-1] = 0.
        weights = np.r_[predicted_with_ones_col, p_ones_row]

        # Solve Kriging systems
        # Each point of unknown area represents different kriging system

        # Transpose points matrix
        transposed = nk.T

        predicted = []

        for idx, point in enumerate(transposed):

            analyzed_pts = unknown_location_points[1][idx, :-1]
            weights = weights.astype(np.float)
            point = point.astype(np.float)

            w = np.linalg.solve(weights, point)

            zhat = self.prepared_data[:, 1].dot(w[:-1])

            if (zhat < 0) or (zhat == np.nan):
                zhat = 0

            point_pop = unknown_location_points[1][idx, -1]
            zhat = (zhat * point_pop) / self.unknown_area_total_population

            # Calculate error

            sigmasq = (w.T * point)[0]
            if sigmasq < 0:
                sigma = 0
            else:
                sigma = np.sqrt(sigmasq)
            predicted.append([zhat, sigma, w[-1], w, analyzed_pts])

        return np.array(predicted)  # return list of lists: [zhat, sigma, mean, weights, point coordinates]

    def calculate_weight_arr(self):
        """Function calculates additional weights for the diagonal matrix of predicted values"""

        vals_of_neigh_areas = self.prepared_data[:, 1]
        pop_of_neigh_areas = self.prepared_data[:, -1]

        weighted = np.sum(vals_of_neigh_areas * pop_of_neigh_areas)
        weights_arr = weighted / np.sum(pop_of_neigh_areas)
        w = np.ones(shape=vals_of_neigh_areas.shape)
        w = (weights_arr * w) / pop_of_neigh_areas
        return np.diag(w)

    def _calculate_avg_semivars_between_known(self, list_of_distances):
        """
        Function calculates average semivariance between known areas
        :param list_of_distances: (numpy array)
            [id base,
                [
                    id other,
                    [base point value, other point value, distance between points]
                ]
            ]
        :return: (numpy array) array of predicted and weighted semivariances
        """

        semivariances = []
        for area in list_of_distances:
            points_group = area[1]
            points_group = [x[1] for x in points_group]
            smv = self.block_to_block_smv.calculate_average_semivariance(points_group)
            semivariances.append(smv)
        return np.array(semivariances)
