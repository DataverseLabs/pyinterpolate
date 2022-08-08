import cProfile
import numpy as np
from pyinterpolate.io.read_data import read_txt
from pyinterpolate.kriging.point_kriging import kriging
from pyinterpolate.variogram.empirical import build_experimental_variogram
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


points = read_txt('../../samples/pl_dem.txt')
pts_size = len(points)
train_size = int(pts_size / 2)
idxs = np.arange(0, pts_size)
train_idxs = np.random.choice(idxs, size=train_size, replace=False)

train_set = points[train_idxs]
test_set = points[~train_idxs]
test_points = test_set[:, :-1]

ex = build_experimental_variogram(train_set, step_size=0.2, max_range=4)
tv = TheoreticalVariogram()
tv.autofit(empirical_variogram=ex)


def krigeme():
    _ = kriging(observations=train_set,
                theoretical_model=tv,
                points=test_points,
                min_no_neighbors=4,
                number_of_workers=4)
    return 0


if __name__ == '__main__':
    cProfile.run('krigeme()', filename='kmultprofile_v0.3.0.profile')
