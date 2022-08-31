import cProfile
import numpy as np
from pyinterpolate.io.read_data import read_txt
from pyinterpolate.kriging.models.point.simple_kriging import simple_kriging
from pyinterpolate.variogram.empirical import build_experimental_variogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


points = read_txt('../../samples/pl_dem.txt')
pts_size = len(points)
train_size = int(pts_size / 2)
idxs = np.arange(0, pts_size)
train_idxs = np.random.choice(idxs, size=train_size, replace=False)

mean = np.mean(points[:, -1])

train_set = points[train_idxs]
test_set = points[~train_idxs]
test_points = test_set[:, :-1]

ex = build_experimental_variogram(train_set, step_size=0.2, max_range=4)
tv = TheoreticalVariogram()
tv.autofit(experimental_variogram=ex)


def krigeme():
    for uloc in test_points:
        _ = simple_kriging(theoretical_model=tv,
                           known_locations=train_set,
                           unknown_location=uloc,
                           min_no_neighbors=4,
                           process_mean=float(mean))
    return 0


if __name__ == '__main__':
    cProfile.run('krigeme()', filename='skprofile_v0.3.0.profile')
