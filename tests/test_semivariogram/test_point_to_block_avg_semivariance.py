import pandas as pd

from pyinterpolate.semivariogram.deconvolution.point_to_block_semivariance import calculate_average_p2b_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram

DATA = [
    [0, 0, 10, 'A', 10, 5],
    [0, 0, 10, 'A', 1, 2],
    [0, 0, 10, 'B', 3, 3],
    [0, 1, 5, 'A', 10, 5],
    [0, 1, 5, 'A', 1, 2],
    [0, 1, 5, 'B', 3, 3],
]

DS = pd.DataFrame(
    data=DATA,
    columns=['x', 'y', 'v', 'idx', 'kv', 'd']
)

PARAMS = {
    'nugget': 0,
    'sill': 7,
    'rang': 3.5,
    'variogram_model_type': 'linear'
}

SEMIVAR = TheoreticalVariogram(model_params=PARAMS)


def test_1():
    estimated = calculate_average_p2b_semivariance(
        ds=DS,
        semivariogram_model=SEMIVAR,
        block_x_coo_col='x',
        block_y_coo_col='y',
        block_val_col='v',
        neighbor_idx_col='idx',
        neighbor_val_col='kv',
        distance_col='d'
    )
    assert len(estimated) == 4
    assert isinstance(estimated, pd.Series)
