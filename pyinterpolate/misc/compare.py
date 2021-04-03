import numpy as np

from pyinterpolate.kriging.areal_poisson_kriging.areal_kriging import ArealKriging
from pyinterpolate.kriging.areal_poisson_kriging.centroid_based.centroid_poisson_kriging \
    import CentroidPoissonKriging
from pyinterpolate.kriging.point_kriging.kriging import Krige


class KrigingComparison:
    """
    Class compares different areal kriging models and techniques.
    """

    def __init__(self, theoretical_semivariogram, areas, points, search_radius, ranges_of_observations,
                 simple_kriging_mean=None, training_set_frac=0.8, iters=20):
        """

        INITIALIZATION PARAMS:

        :param theoretical_semivariogram: (TheoreticalSemivariogram) modeled semivariogram,
        :param areas: (numpy array) areas in the form:
            [area_id, polygon, centroid x, centroid y, value]
        :param points: (numpy array) points within areas in the form:
            [area_id, [point_position_x, point_position_y, value]]
        :param search_radius: (float) minimal search radius to looking for neighbors,
        :param ranges_of_observations: (int) how many neighbors include in analysis,
        :param training_set_frac: (float in range 0-1) how many values set as a "known points",
        :param iters: (int) number of tests (more tests == more accurate Root Mean Squared Error of comparisons).
        """

        self.semivariance = theoretical_semivariogram
        self.areas = areas
        self.points = points
        self.radius = search_radius
        self.ranges = ranges_of_observations
        self.simple_kriging_mean = simple_kriging_mean
        self.frac = training_set_frac
        self.iters = iters

        self.k_types = {
            'PK-ata': 0.,
            'PK-centroid': 0.,
            'OK': 0.,
            'SK': 0.
        }

        self.evaluation_output = {}  # {range: k_types and values}

    def _divide_train_test(self):
        all_ids = self.areas[:, 0]
        training_set_size = int(len(all_ids) * self.frac)
        training_ids = np.random.choice(all_ids,
                                        size=training_set_size,
                                        replace=False)
        training_areas = np.array(
            [a for a in self.areas if a[0] in training_ids]
        )
        test_areas = np.array(
            [a for a in self.areas if a[0] not in training_ids]
        )
        training_pts = np.array(
            [pt for pt in self.points if pt[0] in training_ids]
        )
        test_pts = np.array(
            [pt for pt in self.points if pt[0] not in training_ids]
        )

        output = [training_areas, training_pts,
                  test_areas, test_pts]
        return output

    def _run_pk_ata(self, training_areas, training_points, test_areas, test_points, number_of_obs):
        # Poisson Kriging model Area-to-area

        kriging_model = ArealKriging(semivariogram_model=self.semivariance,
                                     known_areas=training_areas,
                                     known_areas_points=training_points)
        pk_pred = []
        for unknown_area in test_areas:
            unknown_pts = test_points[test_points[:, 0] == unknown_area[0]][0]

            # Predict
            predicted = kriging_model.predict(unknown_pts, number_of_obs, self.radius)
            err = np.sqrt(
                (unknown_area[-1] - predicted[0])**2
            )
            pk_pred.append(err)
        return np.mean(pk_pred)

    def _run_pk_centroid(self, training_areas, training_points, test_areas, test_points, number_of_obs):
        # Poisson Kriging centroid based approach

        kriging_model = CentroidPoissonKriging(semivariogram_model=self.semivariance,
                                               known_areas=training_areas,
                                               known_areas_points=training_points)

        c_pred = []
        for unknown_area in test_areas:
            unknown_pts = test_points[test_points[:, 0] == unknown_area[0]][0]

            # Predict
            try:
                predicted = kriging_model.predict(unknown_area, unknown_pts, number_of_obs,
                                                  self.radius, True)
                err = np.sqrt(
                    (unknown_area[-1] - predicted[0]) ** 2
                )
                c_pred.append(err)
            except ValueError:
                err = unknown_area[-1]
                c_pred.append(err)
        return np.mean(c_pred)

    def _run_ok_point(self, training_areas, test_areas, number_of_obs):
        # Ordinary and Simple Kriging

        kriging_data = training_areas[:, -3:]
        p_kriging = Krige(semivariogram_model=self.semivariance,
                          known_points=kriging_data)

        o_pred = []
        for unknown_area in test_areas:
            unknown_centroids = unknown_area[-3:-1]

            # Predict
            try:
                predicted = p_kriging.ordinary_kriging(unknown_centroids, number_of_obs)
                err = np.sqrt(
                    (unknown_area[-1] - predicted[0]) ** 2
                )
                o_pred.append(err)
            except ValueError:
                err = unknown_area[-1]
                o_pred.append(err)

        return np.mean(o_pred)

    def _run_sk_point(self, training_areas, test_areas, number_of_obs):
        # Ordinary and Simple Kriging

        kriging_data = training_areas[:, -3:]
        p_kriging = Krige(semivariogram_model=self.semivariance,
                          known_points=kriging_data)

        s_pred = []
        for unknown_area in test_areas:
            unknown_centroids = unknown_area[-3:-1]

            # Predict
            try:
                predicted = p_kriging.simple_kriging(unknown_centroids, number_of_obs,
                                                     global_mean=self.simple_kriging_mean)
                err = np.sqrt(
                    (unknown_area[-1] - predicted[0]) ** 2
                )
                s_pred.append(err)
            except ValueError:
                err = unknown_area[-1]
                s_pred.append(err)

        return np.mean(s_pred)

    def run_tests(self):
        """
        Method compares area-to-area, area-to-point and centroid based area Poisson Kriging.
        """
        for number_of_obs in self.ranges:
            pk_evals = []
            ck_evals = []
            ok_evals = []
            sk_evals = []

            for i in range(self.iters):
                # Generate training and test set
                sets = self._divide_train_test()

                # Poisson Kriging
                pk_eval = self._run_pk_ata(sets[0], sets[1], sets[2], sets[3], number_of_obs)
                pk_evals.append(pk_eval)

                # Centroid-based PK
                c_eval = self._run_pk_centroid(sets[0], sets[1], sets[2], sets[3], number_of_obs)
                ck_evals.append(c_eval)

                # Ordinary Kriging
                ok_eval = self._run_ok_point(sets[0], sets[2], number_of_obs)
                ok_evals.append(ok_eval)

                # Simple Kriging
                if self.simple_kriging_mean is None:
                    pass
                else:
                    sk_eval = self._run_sk_point(sets[0], sets[2], number_of_obs)
                    sk_evals.append(sk_eval[1])

            # Mean of values

            pk_rmse = np.mean(pk_evals)
            ck_rmse = np.mean(ck_evals)
            ok_rmse = np.mean(ok_evals)

            # Simple Kriging case
            if self.simple_kriging_mean is None:
                sk_rmse = np.nan
            else:
                sk_rmse = np.mean(sk_evals)

            # Update dict
            d = self.k_types.copy()

            d['PK-ata'] = float(pk_rmse)
            d['PK-centroid'] = float(ck_rmse)
            d['OK'] = float(ok_rmse)
            d['SK'] = float(sk_rmse)

            self.evaluation_output[number_of_obs] = d

            # Inform
            print('Evaluation metrics updated')
            print('Number of ranges:', number_of_obs)
            print('ROOT MEAN SQUARED ERROR VALUES')
            print('Poisson Kriging ATA -', pk_rmse)
            print('Poisson Kriging centroids -', ck_rmse)
            print('Ordinary Kriging -', ok_rmse)
            print('Simple Kriging -', sk_rmse)
