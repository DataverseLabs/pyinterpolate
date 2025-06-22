import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.core.pipelines.interpolate import interpolate_points
from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram

if __name__ == '__main__':
    df = pd.read_csv('dem.csv')
    dem_geometry = gpd.points_from_xy(x=df['longitude'], y=df['latitude'], crs='epsg:4326')
    dem = gpd.GeoDataFrame(df, geometry=dem_geometry)

    # Transform crs to metric values
    dem.to_crs(epsg=2180, inplace=True)
    dem['x'] = dem.geometry.x
    dem['y'] = dem.geometry.y

    exp_var = ExperimentalVariogram(
        dem[['geometry', 'dem']], step_size=500, max_range=10_000
    )

    theo = TheoreticalVariogram()
    theo.autofit(
        experimental_variogram=exp_var
    )

    rx = np.linspace(np.min(dem['x']), np.max(dem['x']), 10000)
    ry = np.linspace(np.min(dem['y']), np.max(dem['y']), 10000)

    rr = np.array(list(zip(rx, ry)))

    ipts = interpolate_points(
        theoretical_model=theo,
        known_locations=dem[['x', 'y', 'dem']].to_numpy(),
        unknown_locations=rr,
        no_neighbors=16
    )
