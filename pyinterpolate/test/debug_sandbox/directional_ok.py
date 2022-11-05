# Imports
from ctypes import Union
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import datetime

from tqdm import tqdm

from pyinterpolate import read_txt
from pyinterpolate import ordinary_kriging  # kriging models
from pyinterpolate.kriging.utils.process import get_predictions, solve_weights
from pyinterpolate.variogram.empirical.experimental_variogram import DirectionalVariogram
from pyinterpolate import TheoreticalVariogram

# Read data
dem = read_txt('../samples/point_data/txt/pl_dem_epsg2180.txt')


def ordinary_kriging2(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location,
        neighbors_range=None,
        no_neighbors=4,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        err_to_nan=False
) -> List:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        A trained theoretical variogram model.

    known_locations : numpy array
        The known locations.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing.

    err_to_nan : bool, default=False
        Return ``NaN`` if algorithm detects singular matrix.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``
    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            no_neighbors,
                                            use_all_neighbors_in_range)

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    while True:
        try:
            output_weights = solve_weights(weights, k, allow_approximate_solutions)
        except np.linalg.LinAlgError as _:
            weights = weights[:-2, :-2]

            p_ones = np.ones((weights.shape[0], 1))
            predicted_with_ones_col = np.c_[weights, p_ones]
            p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
            p_ones_row[0][-1] = 0.
            weights = np.r_[predicted_with_ones_col, p_ones_row]

            k = k[:-2]
            k_ones = np.ones(1)[0]
            k = np.r_[k, k_ones]
        else:
            break

    zhat = dataset[:len(output_weights)-1, -2].dot(output_weights[:-1])

    sigma = np.matmul(output_weights.T, k)

    if sigma < 0:
        return [zhat, np.nan, unknown_location[0], unknown_location[1]]

    return [zhat, sigma, unknown_location[0], unknown_location[1]]

def ordinary_kriging3(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location,
        neighbors_range=None,
        no_neighbors=4,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        err_to_nan=False
) -> List:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        A trained theoretical variogram model.

    known_locations : numpy array
        The known locations.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing.

    err_to_nan : bool, default=False
        Return ``NaN`` if algorithm detects singular matrix.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``
    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            no_neighbors,
                                            use_all_neighbors_in_range)

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    solved = np.linalg.lstsq(weights, k)
    output_weights = solved[0]

    zhat = dataset[:, -2].dot(output_weights[:-1])

    sigma = np.matmul(output_weights.T, k)

    if sigma < 0:
        return [zhat, np.nan, unknown_location[0], unknown_location[1]]

    return [zhat, sigma, unknown_location[0], unknown_location[1]]


def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1), int(frac * len(dataset)), replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


known_points, unknown_points = create_model_validation_sets(dem)
max_range = 10000
step_size = 500

t0 = datetime.datetime.now()
dirvar = DirectionalVariogram(known_points, step_size=step_size, max_range=max_range)
tx = (datetime.datetime.now() - t0).total_seconds()

print(tx)

theo_iso = TheoreticalVariogram()
theo_iso.autofit(experimental_variogram=dirvar.directional_variograms['ISO'],
                 model_types='linear')

k1 = []
k2 = []
k3 = []

for pt in tqdm(unknown_points):
    kr1 = ordinary_kriging(theoretical_model=theo_iso,
                           known_locations=known_points,
                           unknown_location=pt[:-1],
                           no_neighbors=16,
                           err_to_nan=True)
    kr2 = ordinary_kriging2(theoretical_model=theo_iso,
                           known_locations=known_points,
                           unknown_location=pt[:-1],
                           no_neighbors=16,
                           err_to_nan=False)
    kr3 = ordinary_kriging3(theoretical_model=theo_iso,
                           known_locations=known_points,
                           unknown_location=pt[:-1],
                           no_neighbors=16,
                           err_to_nan=False)

    k1.append(kr1)
    k2.append(kr2)
    k3.append(kr3)



def arr2gdf(arr, pred_col, err_col, drop_xy=False):
    gdf = gpd.GeoDataFrame(arr)
    gdf.columns = [pred_col, err_col, 'x', 'y']
    gdf['geometry'] = gpd.points_from_xy(gdf['x'], gdf['y'])

    if drop_xy:
        return gdf[[pred_col, err_col, 'geometry']]
    else:
        return gdf


iso_gdf_1 = arr2gdf(k1, 'iso-pred-0', 'iso-err-0', drop_xy=True)
iso_gdf_2 = arr2gdf(k2, 'iso-pred-while', 'iso-err-while', drop_xy=True)
iso_gdf_3 = arr2gdf(k3, 'iso-pred-lsa', 'iso-err-lsa', drop_xy=True)

df = gpd.GeoDataFrame(unknown_points, columns=['x', 'y', 'dem'])
df['geometry'] = gpd.points_from_xy(df['x'], df['y'])

df = df.merge(iso_gdf_1, on='geometry')
df = df.merge(iso_gdf_2, on='geometry')
df = df.merge(iso_gdf_3, on='geometry')

df.to_csv('test_ok_data/check_results.csv')
