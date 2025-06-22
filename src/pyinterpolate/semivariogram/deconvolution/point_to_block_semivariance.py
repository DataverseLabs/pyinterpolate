from typing import Union, Hashable

import pandas as pd

from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def calculate_average_p2b_semivariance(
        ds: pd.DataFrame,
        semivariogram_model: TheoreticalVariogram,
        block_x_coo_col: Union[Hashable, str],
        block_y_coo_col: Union[Hashable, str],
        block_val_col: Union[Hashable, str],
        neighbor_idx_col: Union[Hashable, str],
        neighbor_val_col: Union[Hashable, str],
        distance_col: Union[Hashable, str],
) -> pd.Series:
    """
    Function calculates average semivariance between a single point from
    the block Pj and all other points from the block Pi.

    Parameters
    ----------
    ds : DataFrame
        DataFrame with columns:
          * block point support coordinate x
          * block point support coordinate y
          * block point support value
          * neighbor index
          * neighboring block point support value
          * distance to the point in neighboring block

    semivariogram_model : TheoreticalVariogram
        Fitted variogram model.

    block_x_coo_col : Union[Hashable, str]
        Column name with block point support coordinate x.

    block_y_coo_col: Union[Hashable, str]
        Column name with block point support coordinate y.

    block_val_col: Union[Hashable, str]
        Column name with block point support value.

    neighbor_idx_col: Union[Hashable, str]
        Column name with neighbor index.

    neighbor_val_col: Union[Hashable, str]
        Column name with neighbor's point support value.

    distance_col: Union[Hashable, str]
        Column with distance to the point in neighboring block.

    Returns
    -------
    p2b : DataFrame
        DataFrame with columns:
          * block point support coordinate x
          * block point support coordinate y
          * neighbor index
          * average semivariance with a neighbor
    """
    ds = ds.copy()

    ds['gamma'] = semivariogram_model.predict(ds[distance_col].to_numpy())
    ds['custom_weights'] = ds[block_val_col] * ds[neighbor_val_col]
    ds['gamma_weights'] = ds['gamma'] * ds['custom_weights']

    # groupby
    gs = ds.groupby(
        [block_x_coo_col, block_y_coo_col, neighbor_idx_col]
    )[['custom_weights', 'gamma_weights']].sum()

    # Calculate weighted semivariance
    weighted_semivariance = gs['gamma_weights'] / gs['custom_weights']

    return weighted_semivariance
