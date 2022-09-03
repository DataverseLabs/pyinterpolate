import cProfile as cp
from pstats import Stats
from pyinterpolate.io_ops import read_point_data
from pyinterpolate.kriging import Krige
from pyinterpolate.semivariance import calculate_semivariance, TheoreticalSemivariogram


if __name__ == '__main__':
    dem = read_point_data('../../../sample_data/point_data/poland_dem_gorzow_wielkopolski', data_type='txt')
    search_radius = 0.01
    max_range = 0.32
    number_of_ranges = 32
    exp_semivar = calculate_semivariance(data=dem, step_size=search_radius, max_range=max_range)
    semivar = TheoreticalSemivariogram(points_array=dem, empirical_semivariance=exp_semivar)
    semivar.find_optimal_model(weighted=False, number_of_ranges=number_of_ranges)
    model = Krige(semivariogram_model=semivar, known_points=dem)
    pr = cp.Profile()
    pr.enable()
    _ = model.ordinary_kriging(dem[1, :-1],
                               min_no_neighbors=16,
                               max_no_neighbors=256,
                               test_anomalies=False)
    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(5)
