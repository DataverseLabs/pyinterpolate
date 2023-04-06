from pyinterpolate import read_txt, build_experimental_variogram, build_theoretical_variogram
from pyinterpolate.validation.cross_validation import validate_kriging


dem = read_txt('../../samples/point_data/txt/pl_dem_epsg2180.txt')

step_radius = 500  # meters
max_range = 10000  # meters

exp_semivar = build_experimental_variogram(input_array=dem, step_size=step_radius, max_range=max_range)
semivar = build_theoretical_variogram(experimental_variogram=exp_semivar,
                                      model_type='linear',
                                      sill=exp_semivar.variance,
                                      rang=10000)

validation_results = validate_kriging(
    dem, theoretical_model=semivar, no_neighbors=16
)

print(validation_results[0])
print(validation_results[1])