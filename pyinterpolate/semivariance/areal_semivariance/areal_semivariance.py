import numpy as np
import matplotlib.pyplot as plt
from pyinterpolate.distance.calculate_distances import calc_block_to_block_distance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram

from pyinterpolate.semivariance.areal_semivariance.within_block_semivariance.calculate_semivariance_within_blocks\
    import calculate_semivariance_within_blocks
from pyinterpolate.semivariance.areal_semivariance.within_block_semivariance.calculate_average_semivariance\
    import calculate_average_semivariance
from pyinterpolate.semivariance.areal_semivariance.block_to_block_semivariance.\
    calculate_block_to_block_semivariance import calculate_block_to_block_semivariance


class ArealSemivariance:
    """
    Class calculates semivariance of areas for Poisson Kriging (area to area and area to point).

    INITIALIZATION PARAMS:

    :param areal_data: (numpy array / list) [area_id, area_geometry, centroid x,
        centroid y, value],
    :param max_areal_range: (float) max distance to perform distance and semivariance calculations,
    :param areal_step_size: (float) step size for search radius,
    :param areal_points_data: (numpy array / list of lists)
        [area_id, [point_position_x, point_position_y, value]]
    :param weighted_semivariance: (bool) if False then each distance is treated equally when calculating
        theoretical semivariance; if True then semivariances closer to the point of origin have more weight,
    :param verbose: (bool) if True then all messages are printed, otherwise nothing.
    """

    def __init__(self, areal_data, areal_step_size, max_areal_range,
                 areal_points_data, weighted_semivariance=False, verbose=False):

        """
        INPUT:

        :param areal_data: (numpy array / list of lists)
            [area_id, area_geometry, centroid coordinate x, centroid coordinate y, value],
        :param areal_step_size: (float) step size for search radius,
        :param max_areal_range: (float) max distance to perform distance and semivariance calculations,
        :param areal_points_data: (numpy array / list of lists)
            [area_id, [point_position_x, point_position_y, value]]
        :param weighted_semivariance: (bool) if False then each distance is treated equally when calculating
            theoretical semivariance; if True then semivariances closer to the point have more weight,
        :param verbose: (bool) if True then all messages are printed, otherwise nothing.
        """

        # Initial data
        self.areal_dataset = areal_data
        self.areal_centroids = self.areal_dataset[:, 2:]
        self.areal_lags = np.arange(0, max_areal_range, areal_step_size)
        self.areal_ss = areal_step_size
        self.areal_max_range = max_areal_range
        self.within_area_points = areal_points_data
        self.weighted_semivariance = weighted_semivariance

        # Semivariogram models
        self.experimental_semivariogram = None
        self.theoretical_semivariance_model = None
        self.inblock_semivariance = None
        self.within_block_semivariogram = None
        self.between_blocks_semivariogram = None
        self.regularized_semivariogram = None

        # Model parameters
        self.distances_between_blocks = None
        self.verbose = verbose

    def _calculate_empirical_semivariance(self):
        """Method calculates theoretical semivariance between areal centroids
        :return gamma: (float) semivariance between areal centroids."""
        gamma = calculate_semivariance(self.areal_centroids,  self.areal_ss, self.areal_max_range)
        return gamma

    def _calculate_theoretical_semivariance(self):
        """Method calculates theoretical semivariogram of a dataset
        :return ts: (TheoreticalSemivariogram) Fitted semivariogram model."""
        ts = TheoreticalSemivariogram(self.areal_centroids, self.experimental_semivariogram)
        ts.find_optimal_model(weighted=self.weighted_semivariance)
        return ts

    # ---------------------------------- AREAL SEMIVARIANCE CALCULATIONS ----------------------------------

    def regularize_semivariogram(self, within_block_semivariogram=None, between_blocks_semivariogram=None,
                                 empirical_semivariance=None, theoretical_semivariance_model=None):
        """
        Function calculates regularized point support semivariogram in the form given in:

        Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units,
            Mathematical Geology 40(1), 101-128, 2008

        Function has the form: gamma_v(h) = gamma(v, v_h) - gamma_h(v, v) where:

        - gamma_v(h) - regularized semivariogram,
        - gamma(v, v_h) - semivariogram value between any two blocks separated by the distance h,
        - gamma_h(v, v) - arithmetical average of within-block semivariogram

        INPUT:

        :param within_block_semivariogram: (numpy array) mean semivariance between the blocks:
            yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:
            y(va, va) and y(va+h, va+h) are the inblock semivariances of block a and block a+h separated
                by the distance h weighted by the inblock population.
        :param between_blocks_semivariogram: (numpy array) semivariance between all blocks calculated from the
            theoretical model,
        :param empirical_semivariance: (numpy array) empirical semivariance between area centroids, default=None, if
            None is provided then empirical semivariance is computed by the _calculate_empirical_semivariance
            method from area centroids,
        :param theoretical_semivariance_model: (TheoreticalSemivariogram) theoretical semivariance model from
            TheoreticalSemivariance class, default is None, if None is provided then theoretical model is derived
            from area centroids and empirical semivariance.

        OUTPUT:

        :return: semivariance: numpy array of pair of lag and semivariance values where:

            - semivariance[0] = array of lags
            - semivariance[1] = array of lag's values
            - semivariance[2] = array of number of points in each lag
        """

        # SET SEMIVARIANCES
        if empirical_semivariance is None:
            self.experimental_semivariogram = self._calculate_empirical_semivariance()
        else:
            self.experimental_semivariogram = empirical_semivariance

        if theoretical_semivariance_model is None:
            self.theoretical_semivariance_model = self._calculate_theoretical_semivariance()
        else:
            self.theoretical_semivariance_model = theoretical_semivariance_model

        # GET WITHIN-AREA SEMIVARIANCE gamma_h(v, v)

        if not within_block_semivariogram:
            self.within_block_semivariogram = self.calculate_mean_semivariance_between_blocks()
        else:
            self.within_block_semivariogram = within_block_semivariogram

        # GET SEMIVARIANCE BETWEEN BLOCKS gamma(v, v_h)

        if not between_blocks_semivariogram:
            self.between_blocks_semivariogram = self.calculate_semivariance_between_areas()
        else:
            self.between_blocks_semivariogram = between_blocks_semivariogram

        regularized_semivariogram = self.between_blocks_semivariogram.copy()
        regularized_semivariogram[:, 1] = (
            self.between_blocks_semivariogram[:, 1] - self.within_block_semivariogram[:, 1]
        )

        # CHECK VALUES BELOW 0

        for idx, row in enumerate(regularized_semivariogram):
            if row[1] < 0:
                regularized_semivariogram[idx, 1] = 0

        self.regularized_semivariogram = regularized_semivariogram.copy()

        return self.regularized_semivariogram

    # ---------------------------------- WITHIN-BLOCK SEMIVARIANCE CALCULATIONS ----------------------------------

    def calculate_mean_semivariance_between_blocks(self, distances=None):
        """
        Function calculates average semivariance between blocks separated by a vector h according to the equation:

        yh(v, v) = 1 / (2*N(h)) SUM(from a=1 to N(h)) [y(va, va) + y(va+h, va+h)], where:

        - y(va, va) and y(va+h, va+h) are estimated according to the function calculate_semivariance_within_blocks,
        - h are estimated according to the block_to_block_distances function.

        INPUT:

        :param distances: if given then this step of calculation is skipped

        OUTPUT:

        :return: [s, d]

                - s - semivariances in the form: list of [[lag, semivariance], [lag_x, semivariance_x], [..., ...]]
                - if distances: d - distances between blocks (dict) in the form: {area_id: {other_area_id: distance,
                    other_area_id: distance,}}
                - else: d = 0.
        """

        # Calculate inblock semivariance
        if self.verbose:
            print('Start of the inblock semivariance calculation')
        # Pass numpy array with [area id, [points within area and their values]] and semivariogram model
        self.inblock_semivariance = np.array(calculate_semivariance_within_blocks(self.within_area_points,
                                                                                  self.theoretical_semivariance_model))
        if self.verbose:
            print('Inblock semivariance calculated successfully')

        # Calculate distance between blocks
        if distances is None:
            if self.verbose:
                print('Distances between blocks: calculation starts')
            self.distances_between_blocks = calc_block_to_block_distance(self.within_area_points)
            if self.verbose:
                print('Distances between blocks have been calculated')
        else:
            if self.verbose:
                print('Distances between blocks are provided, distance skipped, model parameters updated')
            self.distances_between_blocks = distances

        # Calc average semivariance
        avg_semivariance = calculate_average_semivariance(self.distances_between_blocks, self.inblock_semivariance,
                                                          self.areal_lags, self.areal_ss)
        return avg_semivariance

    # ---------------------------------- BLOCK TO BLOCK SEMIVARIANCE CALCULATIONS ----------------------------------

    def calculate_semivariance_between_areas(self):
        """
        Function calculates semivariance between areas based on their division into smaller blocks. It is
            gamma(v, v_h) - semivariogram value between any two blocks separated by the distance h.

        OUTPUT:

        :return semivariogram: (array) Semivariances between blocks for a given lags based on the inblock division into
            smaller blocks. It is numpy array of the form [[lag, semivariance], [next lag, other semivariance], [...]].
        """
        # Check if distances are calculated
        if self.distances_between_blocks is None:
            if self.verbose:
                print('Distances between blocks: calculation starts')
            self.distances_between_blocks = calc_block_to_block_distance(self.within_area_points)
            if self.verbose:
                print('Distances between blocks have been calculated')

        if self.verbose:
            print('Calculation of semivariances between areas separated by chosen lags')
        blocks = calculate_block_to_block_semivariance(self.within_area_points,
                                                       self.distances_between_blocks,
                                                       self.theoretical_semivariance_model)
        if self.verbose:
            print('Semivariance between blocks for a given lags calculated')
            print('Calculation of the mean semivariance for a given lag')
        semivariogram = []

        for areal_lag in self.areal_lags:
            dists_and_semivars = blocks[0].copy().flatten()
            smvs = []
            for dists_sem_group in dists_and_semivars:
                dists_sem_group = np.array(dists_sem_group)

                # Get dists greater than areal lag minus step size
                lower_limit = areal_lag - self.areal_ss
                upper_limit = areal_lag + self.areal_ss
                chosen_dists_upper_limit = dists_sem_group[dists_sem_group[:, 0] > lower_limit]

                # Get dists lower than areal lag plus step size
                chosen_dists = chosen_dists_upper_limit[chosen_dists_upper_limit[:, 0] < upper_limit]

                if len(chosen_dists[:, 1]) != 0:
                    semivars_inside = np.mean(chosen_dists[:, 1])
                else:
                    semivars_inside = 0
                smvs.append(semivars_inside)

            if np.sum(smvs) == 0:
                semivariogram.append([areal_lag, 0])
            else:
                smvs = np.mean(smvs)
                semivariogram.append([areal_lag, smvs])

        if self.verbose:
            print('End of block to block semivariogram calculation')
        return np.array(semivariogram)

    def show_semivariograms(self):
        """
        Function shows semivariograms calculated by the class: Empirical semivariogram, Theoretical model,
        Inblock Semivariance, Within-block semivariogram, Between blocks semivariogram, Regularized output.
        """
        plt.figure(figsize=(12, 12))
        plt.plot(self.experimental_semivariogram[:, 0], self.experimental_semivariogram[:, 1], 'bo')
        plt.plot(self.experimental_semivariogram[:, 0],
                 self.theoretical_semivariance_model.predict(self.experimental_semivariogram[:, 0]),
                 color='black', linestyle='dotted')
        mean_inblock_semivar = np.mean(self.inblock_semivariance[:, 1])
        inblock_sem = np.ones((1, len(self.experimental_semivariogram[:, 0]))) * mean_inblock_semivar
        plt.plot(self.experimental_semivariogram[:, 0], inblock_sem[0],
                 color='yellow', linestyle='-')
        plt.plot(self.within_block_semivariogram[:, 0], self.within_block_semivariogram[:, 1], color='green')
        plt.plot(self.between_blocks_semivariogram[:, 0], self.between_blocks_semivariogram[:, 1], color='pink')
        plt.plot(self.regularized_semivariogram[:, 0], self.regularized_semivariogram[:, 1], color='red')

        plt.legend(['Empirical semivariogram', 'Theoretical model', 'Inblock Semivariance',
                    'Within-block semivariogram', 'Between blocks semivariogram', 'Regularized output'])
        plt.title('Areal semivariograms')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()
