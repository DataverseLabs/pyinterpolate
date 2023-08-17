# Issue: https://github.com/DataverseLabs/pyinterpolate/issues/213
from pyinterpolate import read_txt, build_experimental_variogram


DEM = '../samples/point_data/txt/pl_dem.txt'
SEARCH_RADIUS = 0.1
MAX_RANGE = 0.32
DIRECTION = 150
TOLERANCE = 0.1


if __name__ == '__main__':
    ds = read_txt(DEM)
    exp_var = build_experimental_variogram(ds,
                                           step_size=SEARCH_RADIUS,
                                           max_range=MAX_RANGE,
                                           direction=DIRECTION,
                                           tolerance=TOLERANCE,
                                           method='e')
