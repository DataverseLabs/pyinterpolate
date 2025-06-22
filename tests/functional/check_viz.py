import numpy as np

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.viz.raster import interpolate_raster
from tests.test_core.sample_data.dataprep import CANCER_DATA

BLOCKS = Blocks(**CANCER_DATA)
DS = BLOCKS.representative_points_array()


if __name__ == '__main__':
    raster = interpolate_raster(
        data=DS,
        dim=100,
        number_of_neighbors=8
    )

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.scatterplot(
        x=DS[:, 0], y=DS[:, 1], hue=DS[:, 2]
    )
    plt.show()

    plt.figure()
    plt.imshow(raster['result'], cmap='plasma')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(raster['error'])
    plt.colorbar()
    plt.show()

    print(np.mean(raster['result']), np.std(raster['result']))
