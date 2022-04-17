from typing import Union, List, Tuple

import os
import numpy as np
import dask

from pyinterpolate.kriging.models.ordinary_kriging import ordinary_kriging
from pyinterpolate.kriging.models.simple_kriging import simple_kriging
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


def kriging(observations: np.ndarray,
            theoretical_model: TheoreticalVariogram,
            points: Union[np.ndarray, List, Tuple],
            how: str = 'ok',
            neighbors_range: Union[float, None] = None,
            min_no_neighbors: int = 1,
            max_no_neighbors: int = -1,
            process_mean: Union[float, None] = None,
            number_of_workers: int = -1) -> np.ndarray:

    if number_of_workers == -1:
        core_num = os.cpu_count()
        if core_num > 1:
            number_of_workers = int(core_num / 2)
        else:
            number_of_workers = core_num

    models = {'ok': ordinary_kriging,
              'sk': simple_kriging}

    if how not in list(models.keys()):
        raise KeyError(f'Given model not available, choose one from {list(models.keys())} instead.')
    else:
        model = models[how]

    results = []

    for point in points:
        prediction = [np.nan, np.nan, np.nan, np.nan]
        if how == 'ok':
            prediction = dask.delayed(model)(
                theoretical_model,
                observations,
                point,
                neighbors_range=neighbors_range,
                min_no_neighbors=min_no_neighbors,
                max_no_neighbors=max_no_neighbors
            )
        elif how == 'sk':
            prediction = model(
                theoretical_model,
                observations,
                point,
                process_mean,
                neighbors_range=neighbors_range,
                min_no_neighbors=min_no_neighbors,
                max_no_neighbors=max_no_neighbors
            )
        results.append(prediction)
    predictions = dask.delayed()(results)
    predictions = np.array(predictions.compute(num_workers=number_of_workers))
    return np.array(predictions)


if __name__ == '__main__':
    # Import data for tests
    from datetime import datetime
    from sample_data.data import SampleData
    from pyinterpolate.io.read_data import read_csv
    from pyinterpolate.variogram.empirical.experimental_variogram import build_experimental_variogram
    from pyinterpolate.variogram.theoretical.semivariogram import build_theoretical_variogram

    sd = SampleData()
    dem = read_csv(sd.dem, val_col_name='dem', lat_col_name='latitude', lon_col_name='longitude')

    # Create variogram
    experimental_model = build_experimental_variogram(dem, 0.01, 0.2)
    theoretical_model = build_theoretical_variogram(experimental_model, 'spherical', sill=500, rang=0.125)

    # Krige
    gridx = np.linspace(np.min(dem[:, 0]), np.max(dem[:, 0]), 100)
    gridy = np.linspace(np.min(dem[:, 1]), np.max(dem[:, 1]), 100)

    grid = np.column_stack([gridx, gridy])

    t0 = datetime.now()
    c = kriging(observations=dem, theoretical_model=theoretical_model, points=grid)
    tx = datetime.now()
    tdelta_dask = (tx - t0).seconds
    print(tdelta_dask)
