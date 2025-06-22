import numpy as np
from pyinterpolate.semivariogram.experimental.functions.directional import \
    from_ellipse_non_weighted


def profile_select_in_ellipse():
    points = np.random.rand(500, 3)
    step_size = 0.05
    max_range = 0.6

    _ = from_ellipse_non_weighted(
        points=points,
        lags=np.linspace(step_size, max_range, 10),
        direction=35,
        tolerance=0.1
    )


if __name__ == '__main__':
    profile_select_in_ellipse()
