import numpy as np
from pyinterpolate.semivariogram.experimental.functions.directional import \
    directional_weighted_semivariance


def profile_select_in_ellipse():
    points = np.random.rand(500, 3)
    weights = np.random.randint(1, 100, len(points))
    step_size = 0.05
    max_range = 0.6

    _ = directional_weighted_semivariance(
        points=points,
        lags=np.linspace(step_size, max_range, 10),
        custom_weights=weights,
        direction=35,
        tolerance=0.1
    )


if __name__ == '__main__':
    profile_select_in_ellipse()
