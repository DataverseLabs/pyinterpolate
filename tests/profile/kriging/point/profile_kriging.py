import cProfile
import numpy as np
from pyinterpolate.io.read_data import read_txt
from pyinterpolate.kriging.point_kriging import kriging
from pyinterpolate.variogram.empirical import build_experimental_variogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


points = read_txt('../../samples/pl_dem_epsg2180.txt')
pts_size = len(points)
train_size = int(pts_size * 0.5)
idxs = np.arange(0, pts_size)
train_idxs = np.random.choice(idxs, size=train_size, replace=False)

train_set = points[train_idxs]
test_set = points[~train_idxs]
test_points = test_set[:, :-1]

ex = build_experimental_variogram(train_set, step_size=1_000, max_range=10_000)
tv = TheoreticalVariogram()
tv.autofit(experimental_variogram=ex, model_types=['circular'])


def krigeme():
    predictions = kriging(observations=train_set,
                          theoretical_model=tv,
                          points=test_points,
                          no_neighbors=4,
                          use_all_neighbors_in_range=False,
                          number_of_workers=1)
    mse = np.mean((predictions[:, 0] - test_points[:, -1])**2)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == '__main__':
    cProfile.run('krigeme()', filename='kmultprofile_v0.3.0.clean.profile')
