from pyinterpolate import read_txt
from pyinterpolate.variogram.indicator.indicator_variogram import ExperimentalIndicatorVariogram, IndicatorVariograms

dem = read_txt('../../samples/point_data/txt/pl_dem_epsg2180.txt')

step_radius = 500  # meters
_max_range = 10000  # meters

ind_variogram = ExperimentalIndicatorVariogram(input_array=dem,
                                               number_of_thresholds=10,
                                               step_size=step_radius,
                                               max_range=_max_range)

ind_vars = IndicatorVariograms(experimental_indicator_variogram=ind_variogram)
ind_vars.fit(
    model_type='basic',
    verbose=False
)
ind_vars.show(subplots=False)
