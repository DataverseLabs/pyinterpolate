import os

import numpy as np

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def build_armstrong_ds():
    ds = get_armstrong_data()
    armstrong_variogram_omnidirectional = ExperimentalVariogram(
        ds,
        step_size=1,
        max_range=6)

    armstrong_variogram_directional = ExperimentalVariogram(
        ds,
        step_size=1.2,
        max_range=6,
        direction=135,
        tolerance=0.02
    )

    variogram_omni = TheoreticalVariogram()
    variogram_omni.autofit(experimental_variogram=armstrong_variogram_omnidirectional)

    variogram_dir = TheoreticalVariogram()
    variogram_dir.autofit(experimental_variogram=armstrong_variogram_directional)

    output = {
        'ds': ds,
        'theo_omni': variogram_omni,
        'theo_dir': variogram_dir
    }
    return output


def build_random_ds():
    ds = np.random.random(size=(1000, 3))
    random_variogram_omnidirectional = ExperimentalVariogram(
        ds,
        step_size=0.05,
        max_range=0.6)

    random_variogram_directional = ExperimentalVariogram(
        ds,
        step_size=0.08,
        max_range=0.65,
        direction=30,
        tolerance=0.02
    )

    variogram_omni = TheoreticalVariogram()
    variogram_omni.autofit(experimental_variogram=random_variogram_omnidirectional)

    variogram_dir = TheoreticalVariogram()
    variogram_dir.autofit(experimental_variogram=random_variogram_directional)

    output = {
        'ds': ds,
        'theo_omni': variogram_omni,
        'theo_dir': variogram_dir
    }
    return output


def build_zeros_ds():
    ds = np.zeros(shape=(1000, 3))
    variogram_omni = TheoreticalVariogram()
    variogram_omni.nugget = 0
    variogram_omni.sill = 0
    variogram_omni.rang = 0
    variogram_omni.model_type = 'linear'
    output = {
        'ds': ds,
        'theo_omni': variogram_omni
    }
    return output


# TODO is this function used?
def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1), int(frac * len(dataset)), replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


def get_armstrong_data():
    my_dir = os.path.dirname(__file__)
    filename = 'armstrong_data.npy'
    filepath = f'{filename}'
    path_to_the_data = os.path.join(my_dir, filepath)
    arr = np.load(path_to_the_data)
    return arr


