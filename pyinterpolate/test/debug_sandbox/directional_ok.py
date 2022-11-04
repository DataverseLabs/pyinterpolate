# Imports
import geopandas as gpd
import numpy as np
import datetime

from pyinterpolate import inverse_distance_weighting  # function for idw
from pyinterpolate import read_txt
from pyinterpolate import build_experimental_variogram  # experimental semivariogram
from pyinterpolate import build_theoretical_variogram  # theoretical models
from pyinterpolate import kriging  # kriging models
from pyinterpolate.variogram.empirical.experimental_variogram import DirectionalVariogram
from pyinterpolate import interpolate_raster
from pyinterpolate import TheoreticalVariogram


# Read data
dem = read_txt('../samples/point_data/txt/pl_dem_epsg2180.txt')


def create_model_validation_sets(dataset: np.array, frac=0.1):
    removed_idx = np.random.randint(0, len(dataset)-1, size=int(frac * len(dataset)))
    training_set = dataset[removed_idx]
    validation_set = np.delete(dataset, removed_idx, 0)
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

kriged_iso = kriging(observations=known_points,
                 theoretical_model=theo_iso,
                 points=unknown_points[:, :-1],
                 no_neighbors=64,
                 err_to_nan=True,
                 use_all_neighbors_in_range=False
                )


def arr2gdf(arr, pred_col, err_col, drop_xy=False):
    gdf = gpd.GeoDataFrame(arr)
    gdf.columns = [pred_col, err_col, 'x', 'y']
    gdf['geometry'] = gpd.points_from_xy(gdf['x'], gdf['y'])

    if drop_xy:
        return gdf[[pred_col, err_col, 'geometry']]
    else:
        return gdf


iso_gdf = arr2gdf(kriged_iso, 'iso-pred', 'iso-err', drop_xy=True)

df = gpd.GeoDataFrame(unknown_points, columns = ['x', 'y', 'dem'])
df['geometry'] = gpd.points_from_xy(df['x'], df['y'])

df = df.merge(iso_gdf, on='geometry')

df['iso-rmse'] = np.sqrt((df['dem'] - df['iso-pred'])**2)

