import numpy as np
import pandas as pd
from tqdm import tqdm

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.kriging.block.centroid_based_poisson_kriging import centroid_poisson_kriging
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from tests.test_semivariogram.sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

THEO = TheoreticalVariogram()
THEO.from_json('regularized.json')


def check_cpk():
    indexes = BLOCKS.block_indexes

    boolean_choice = [True, False]

    ds = []
    for idx in tqdm(indexes):
        for is_weighted_by_point_support in boolean_choice:
            for number_of_neighbors in [8, 16, 32]:
                for neighbors_range in [180_000, 240_000, 400_000]:
                    for use_all_neighbors_in_range in boolean_choice:
                        try:
                            cpk = centroid_poisson_kriging(
                                semivariogram_model=THEO,
                                point_support=PS,
                                unknown_block_index=idx,
                                number_of_neighbors=number_of_neighbors,
                                neighbors_range=neighbors_range,
                                is_weighted_by_point_support=is_weighted_by_point_support,
                                use_all_neighbors_in_range=use_all_neighbors_in_range,
                                raise_when_negative_prediction=False,
                                raise_when_negative_error=False
                            )
                        except ValueError:
                            cpk = {
                                "block_id": idx,
                                "zhat": np.nan,
                                "sig": np.nan
                            }
                        cpk["is_weighted_by_point_support"] = is_weighted_by_point_support
                        cpk["use_all_neighbors_in_range"] = use_all_neighbors_in_range
                        cpk["number_of_neighbors"] = number_of_neighbors
                        cpk["neighbors_range"] = neighbors_range
                        cpk["real"] = BLOCKS.ds[BLOCKS.ds[BLOCKS.index_column_name] == idx][BLOCKS.value_column_name].iloc[0]
                        ds.append(cpk)

    df = pd.DataFrame(ds)
    return df


if __name__ == '__main__':
    data = check_cpk()
    data.to_csv('cpk_output.csv', index=None)
