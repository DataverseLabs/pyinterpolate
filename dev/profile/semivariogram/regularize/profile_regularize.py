from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.deconvolution.regularize import Deconvolution
from dev.profile.semivariogram.calculate_average_semivariance.dataprep import (CANCER_DATA_WITH_CENTROIDS,
                                                                               POINT_SUPPORT_DATA)


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column']
    )

MAX_RANGE = 400000
STEP_SIZE = 20000


if __name__ == '__main__':
    dcv = Deconvolution(verbose=True)
    dcv.fit(blocks=BLOCKS,
            point_support=PS,
            nugget=0.0,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE)

    transformed = dcv.transform(
        max_iters=10,
        limit_deviation_ratio=0.1,
        minimum_deviation_decrease=0.01,
        reps_deviation_decrease=2
    )
    dcv.plot_variograms()
    dcv.plot_weights_change(
        averaged=True
    )
    dcv.plot_weights_change(averaged=False)
    dcv.plot_deviation_change()
