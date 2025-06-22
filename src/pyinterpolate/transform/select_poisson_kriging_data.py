from typing import Union, Hashable

from pyinterpolate.core.data_models.centroid_poisson_kriging import CentroidPoissonKrigingInput
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.poisson_kriging import PoissonKrigingInput
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


# TODO u block centroid and point support as another way of pk
def select_centroid_poisson_kriging_data(
        block_index: Union[str, Hashable],
        point_support: PointSupport,
        semivariogram_model: TheoreticalVariogram,
        number_of_neighbors: int,
        weighted: bool,
        neighbors_range: float = None
) -> CentroidPoissonKrigingInput:
    """
    Function prepares data for the centroid-based Poisson Kriging Process.

    Parameters
    ----------
    block_index : Union[str, Hashable]
        Index of unknown block, then it will be used to take unknown block
        geometry from Blocks, and unknown block's point support from
        the PointSupport.

    point_support : PointSupport
        Point support of polygons.

    semivariogram_model : TheoreticalVariogram
        Regularized semivariogram.

    number_of_neighbors : int
         The minimum number of neighboring blocks.

    weighted : bool
        Are distances between blocks weighted by point support values?

    neighbors_range : float, optional
        The maximum range where other blocks are considered as the neighbors.
        If not provided then function uses semivariogram max range.

    Returns
    -------
    dataset : numpy array
        coordinate x, coordinate y, value, distance to unknown block centroid,
        angles, aggregated point support sum
    """
    # Prepare Kriging data
    # [x, y, value, distance to unknown centroid,
    # difference between angles, block id]
    pk_input = CentroidPoissonKrigingInput(
        block_id=block_index,
        point_support=point_support,
        semivariogram_model=semivariogram_model,
        blocks_indexes=point_support.blocks.block_indexes,
        weighted=weighted
    )

    if neighbors_range is None:
        neighbors_range = semivariogram_model.rang

    if pk_input.is_directional:
        pk_input.select_neighbors_directional(
            max_range=neighbors_range,
            min_number_of_neighbors=number_of_neighbors
        )
    else:
        pk_input.select_neighbors(
            max_range=neighbors_range,
            min_number_of_neighbors=number_of_neighbors
        )
    return pk_input


def select_poisson_kriging_data(block_index: Union[str, Hashable],
                                point_support: PointSupport,
                                semivariogram_model: TheoreticalVariogram,
                                number_of_neighbors: int,
                                neighbors_range: float = None):
    # Prepare Kriging data
    # [unknown x, unknown y, unknown point support value,
    #  other x, other y, other point support value,
    #  distance to other,
    #  angular difference,
    #  block id]

    pk_input = PoissonKrigingInput(
        block_id=block_index,
        point_support=point_support,
        semivariogram_model=semivariogram_model,
        blocks_indexes=point_support.blocks.block_indexes
    )

    if neighbors_range is None:
        neighbors_range = semivariogram_model.rang

    _ = pk_input.select_block_neighbors(
        point_support=point_support,
        max_range=neighbors_range,
        no_closest_neighbors=number_of_neighbors,
        max_tick=45,
    )

    return pk_input
