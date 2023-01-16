import cProfile
import numpy as np

from pyinterpolate.variogram.theoretical.models.variogram_models import cubic_model


lags = np.linspace(20000, 100000, 100000)


def profile_cubic():
    for _ in range(0, 10000):
        _ = cubic_model(lags, 0, 100.01, 2000.41235)
    return 0


if __name__ == '__main__':
    cProfile.run('profile_cubic()', filename='cubic_v0.3.0.profile')
