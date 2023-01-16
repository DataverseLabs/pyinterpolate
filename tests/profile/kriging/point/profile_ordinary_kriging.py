import cProfile
import numpy as np
from pyinterpolate.io.read_data import read_txt
from pyinterpolate.kriging.models.point.ordinary_kriging import ordinary_kriging
from pyinterpolate.variogram.empirical import build_experimental_variogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


points = read_txt('../../samples/pl_dem_epsg2180.txt')
pts_size = len(points)
train_size = int(pts_size / 2)
idxs = np.arange(0, pts_size)
train_idxs = np.random.choice(idxs, size=train_size, replace=False)

train_set = points[train_idxs]
test_set = points[~train_idxs]
test_points = test_set[:, :-1]

ex = build_experimental_variogram(train_set, step_size=1_000, max_range=10_000)
tv = TheoreticalVariogram()
tv.autofit(experimental_variogram=ex)


def krigeme():
    for uloc in test_points:
        _ = ordinary_kriging(theoretical_model=tv,
                             known_locations=train_set,
                             unknown_location=uloc,
                             no_neighbors=4)
    return 0


if __name__ == '__main__':
    cProfile.run('krigeme()', filename='okprofile_v0.3.0.profile')
