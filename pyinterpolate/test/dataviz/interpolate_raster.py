import os
import numpy as np
import matplotlib.pyplot as plt

from pyinterpolate.variogram import build_experimental_variogram, build_theoretical_variogram
from pyinterpolate.viz.raster import interpolate_raster


def get_armstrong_data():
    my_dir = os.path.dirname(__file__)
    filename = 'armstrong_data.npy'
    filepath = f'../samples/point_data/numpy/{filename}'
    path_to_the_data = os.path.abspath(os.path.join(my_dir, filepath))
    arr = np.load(path_to_the_data)
    return arr


def prepare_test_data():
    dataset = get_armstrong_data()
    # Experimental variogram
    exp_var = build_experimental_variogram(dataset, step_size=1, max_range=6)
    # Theoretical variogram
    theo_var = build_theoretical_variogram(exp_var, 'linear', exp_var.variance, 7, 0)
    return dataset, theo_var


def show_data(data_matrix):
    plt.figure()
    plt.imshow(data_matrix, cmap='Spectral_r')
    plt.title('Interpolated dataset')
    plt.colorbar()
    plt.show()


test_data, test_variogram = prepare_test_data()
interpolated = interpolate_raster(data=test_data, dim=100, semivariogram_model=test_variogram)

show_data(interpolated['result'])
show_data(interpolated['error'])
