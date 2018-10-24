import math
import numpy as np


def prepare_min_distance(diffs):
    if diffs[0] < diffs[1]:
        return 0.01 * diffs[0]
    else:
        return 0.01 * diffs[1]


def create_raster(points_list):
    xmax, ymax = points_list.max(axis=0)
    xmin, ymin = points_list.min(axis=0)
    x_range = xmax - xmin
    y_range = ymax - ymin
    differences = np.array([x_range, y_range])
    min_pixel_distance = prepare_min_distance(differences)
    res_x = int(math.ceil(differences[0]/min_pixel_distance))
    res_y = int(math.ceil(differences[1]/min_pixel_distance))
    raster = np.zeros((res_y, res_x))
    return raster, xmin, xmax, ymin, ymax, min_pixel_distance


if __name__ == '__main__':
    from pyinterpolate.kriging.helper_functions.read_data import read_data
    pts = read_data('../../../data/gorzow_dem_10p.dat', sep=',')

    z = create_raster(pts[:, :-1])
    print(z[2] - z[1], z[4] - z[3])