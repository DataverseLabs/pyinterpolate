import numpy as np
from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance


def block_pair_semivariance(block_a, block_b, semivariogram_model):
    """
    Function calculates average semivariance between two blocks based on the counts inside the block.
    :param block_a: block A points in the form of array [[point x1A, point y1A, value v1A],
                                                         [point x2A, point y2A, value v2A],
                                                         [...]
                                                         [point xnA, point ynA, value vnA]]
        All coordinates from the array must be placed inside the block!
    :param block_b: block B points in the form of array [[point x1B, point y1B, value v1B],
                                                         [point x2B, point y2B, value v2B],
                                                         [...]
                                                         [point xnB, point ynB, value vnB]]
        All coordinates from the array must be placed inside the block!
    :param semivariogram_model: (TheoreticalSemivariogram) theoretical semivariance model from TheoreticalSemivariance
        class. Model must be fitted and calculated.
    :return semivariance_mean: (float) Average semivariance between blocks divided by points.
    """
    distances_between_points = calc_point_to_point_distance(block_a, block_b).flatten()

    semivariances = []
    for dist in distances_between_points:
        semivariances.append(
            semivariogram_model.predict(dist)
        )

    semivariance_mean = np.sum(semivariances) / len(semivariances)

    return semivariance_mean


def calculate_block_to_block_semivariance(points_within_area, distances_between_blocks, semivariogram_model):
    """
    Function calculates semivariances between all blocks passed into it based on the points (rectangles) and their
    values inside the blocks.

    :param points_within_area: (numpy array) with area id and points and respective values inside area:
        [area id, [
                    [point x, point y, value] ...
                  ]
        ]
    :param distances_between_blocks: distances_arrays: (arrays)
        array[0] - list of distances between blocks,
        array[1] - list of blocks ids.

        array[0]: [
                    distances[id A to id A, id A to id B, id A to id N],
                    distances[id B to id A, id B to id B, id B to id N],
                    distances[id N to id A, id N to id B, id N to id N]
                ],
        array[1]: [id A, id B, id N]
    :param semivariogram_model: (TheoreticalSemivariogram) Theoretical Semivariogram object,
    :return output_array: (numpy array) semivariances and distances array:
        output_array[0] - list of distances and semivariances between blocks,
        output_array[1] - list of blocks ids.

        output_array[0]: [[[distance between A and A, semivariance between A and A],
                          [distanace between A and B, semivariance between A and B]],

                          [[distance between B and A, semivariance between B and A],
                          [distanace between B and B, semivariance between B and B]],

                          ...]
        output_array[1]: [id A, id B, id ...]
    """
    bb_semivariances = []

    blocks_ids = distances_between_blocks[1]

    if type(points_within_area) == list:
        points_within_area = np.array(points_within_area)

    for first_idx, first_block_id in enumerate(blocks_ids):
        block_to_block_semivariance = []
        for second_idx, second_block_id in enumerate(blocks_ids):
            if first_block_id == second_block_id:
                block_to_block_semivariance.append([0, 0])
            else:
                # Select distance from the first selected block to the second selected block
                distance = distances_between_blocks[0][first_idx, second_idx]

                # Select coordinates of the block centroids
                first_block_points = points_within_area[points_within_area[:, 0] == first_block_id]
                second_block_points = points_within_area[points_within_area[:, 0] == second_block_id]

                # Calculate semivariance between blocks
                semivariance = block_pair_semivariance(first_block_points[0][1], second_block_points[0][1],
                                                       semivariogram_model)
                block_to_block_semivariance.append([distance, semivariance])
        bb_semivariances.append(block_to_block_semivariance)

    output_array = np.array([bb_semivariances, blocks_ids])
    return output_array
