import cProfile
import numpy as np

from pyinterpolate.variogram.theoretical.models.variogram_models import circular_model


lags = np.linspace(20000, 100000, 100000)


def profile_circular():
    for _ in range(0, 1000):
        out = circular_model(lags, 0, 100.01, 2000.41235)
    return out


if __name__ == '__main__':
    cProfile.run('profile_circular()', filename='circular_power_changed_v0.3.0.profile')
