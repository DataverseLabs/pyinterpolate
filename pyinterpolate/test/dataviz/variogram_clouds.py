import os
import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.io import read_txt
from pyinterpolate.variogram.empirical.variogram_cloud import VariogramCloud


def test_with_armstrong_data():

    def get_armstrong_data():
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.abspath(os.path.join(my_dir, filepath))
        arr = np.load(path_to_the_data)
        return arr

    ss = 1.5
    rng = 7
    data = get_armstrong_data()
    vc = VariogramCloud(input_array=data, step_size=ss, max_range=rng)
    vc.plot('box')
    vc.plot('scatter')
    vc.plot('violin')


if __name__ == '__main__':
    test_with_armstrong_data()
    dem = read_txt('../samples/point_data/txt/pl_dem_epsg2180.txt')
    dem = dem[:100, :]
    distances = calc_point_to_point_distance(dem[:, :-1])
    maximum_range = np.max(distances) / 4
    steps = 16
    step_size = maximum_range / steps
    vc = VariogramCloud(input_array=dem, step_size=step_size, max_range=maximum_range)
    vc.plot('box')
    vc.plot('scatter')
    vc.plot('violin')
