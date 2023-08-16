import numpy as np
import pandas as pd
from tqdm import tqdm

from pyinterpolate import build_experimental_variogram, Blocks, PointSupport, Deconvolution, TheoreticalVariogram, \
    area_to_area_pk, area_to_point_pk
from pyinterpolate.pipelines.deconvolution import smooth_area_to_point_pk

DATASET = '../../samples/regularization/cancer_data.gpkg'
OUTPUT = '../../samples/regularization/regularized_directional_variogram.json'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 8


blocks = Blocks()
blocks.from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)

point_support = PointSupport()
point_support.from_files(point_support_data_file=DATASET,
                         blocks_file=DATASET,
                         point_support_geometry_col=GEOMETRY_COL,
                         point_support_val_col=POP10,
                         blocks_geometry_col=GEOMETRY_COL,
                         blocks_index_col=POLYGON_ID,
                         use_point_support_crs=True,
                         point_support_layer_name=POPULATION_LAYER,
                         blocks_layer_name=POLYGON_LAYER)


# Check experimental semivariogram of areal data - this cell may be run multiple times
# before you find optimal parameters

maximum_range = 400000
step_size = 40000

# dt = blocks.data[[blocks.cx, blocks.cy, blocks.value_column_name]]  # x, y, val
# exp_semivar = build_experimental_variogram(input_array=dt, step_size=step_size, max_range=maximum_range,
#                                            direction=45, tolerance=0.1)
#
# # Plot experimental semivariogram
#
# exp_semivar.plot()
#
# reg_mod = Deconvolution(verbose=True)
#
# reg_mod.fit(agg_dataset=blocks,
#             point_support_dataset=point_support,
#             agg_step_size=step_size,
#             agg_max_range=maximum_range,
#             agg_direction=45,
#             agg_tolerance=0.1,
#             variogram_weighting_method='closest',
#             model_types='basic')
#
# reg_mod.transform(max_iters=15)
#
# reg_mod.plot_deviations()
# reg_mod.plot_weights()
# reg_mod.plot_variograms()
# reg_mod.export_model(OUTPUT)

def select_unknown_blocks_and_ps(areal_input, point_support, block_id):
    ar_x = areal_input.cx
    ar_y = areal_input.cy
    ar_val = areal_input.value_column_name
    ps_val = point_support.value_column
    ps_x = point_support.x_col
    ps_y = point_support.y_col
    idx_col = areal_input.index_column_name

    areal_input = areal_input.data.copy()
    point_support = point_support.point_support.copy()

    sample_key = np.random.choice(list(point_support[block_id].unique()))

    unkn_ps = point_support[point_support[block_id] == sample_key][[ps_x, ps_y, ps_val]].values
    known_poses = point_support[point_support[block_id] != sample_key]
    known_poses.rename(columns={
        ps_x: 'x', ps_y: 'y', ps_val: 'ds', idx_col: 'index'
    }, inplace=True)

    unkn_area = areal_input[areal_input[block_id] == sample_key][[idx_col, ar_x, ar_y, ar_val]].values
    known_areas = areal_input[areal_input[block_id] != sample_key]
    known_areas.rename(columns={
        ar_x: 'centroid.x', ar_y: 'centroid.y', ar_val: 'ds', idx_col: 'index'
    }, inplace=True)

    return known_areas, known_poses, unkn_area, unkn_ps

# Read semivariogram
variogram = TheoreticalVariogram()
variogram.from_json(OUTPUT)

# Perform ATA
areas = []

rmse_ata = []
rmse_atp = []
rmse_centroid = []
err_ata = []
err_atp = []
err_centroid = []
diff_ata_atp = []
diff_err_ata_atp = []

# for _ in tqdm(range(200)):
#
#     AREAL_INP, PS_INP, UNKN_AREA, UNKN_PS = select_unknown_blocks_and_ps(blocks,
#                                                                          point_support,
#                                                                          POLYGON_ID)
#
#     if UNKN_AREA[0][0] in areas:
#         continue
#     else:
#         areas.append(UNKN_AREA[0][0])
#
#         uar = UNKN_AREA[0][:-1]
#         uvar = UNKN_AREA[0][-1]
#
#         pk_output_ata = area_to_area_pk(semivariogram_model=variogram,
#                                          blocks=AREAL_INP,
#                                          point_support=PS_INP,
#                                          unknown_block=uar,
#                                          unknown_block_point_support=UNKN_PS,
#                                          number_of_neighbors=NN,
#                                          raise_when_negative_error=False,
#                                          raise_when_negative_prediction=False,
#                                          log_process=False)
#
#         pk_output_atp = area_to_point_pk(semivariogram_model=variogram,
#                                          blocks=AREAL_INP,
#                                          point_support=PS_INP,
#                                          unknown_block=uar,
#                                          unknown_block_point_support=UNKN_PS,
#                                          number_of_neighbors=NN,
#                                          raise_when_negative_error=False,
#                                          raise_when_negative_prediction=False)
#
#         ata_pred = pk_output_ata[1]
#         ata_err = pk_output_ata[2]
#
#         atp_pred = np.sum([x[1] for x in pk_output_atp])
#         atp_err = np.mean([x[2] for x in pk_output_atp])
#
#         rmse_ata.append(np.sqrt((ata_pred - uvar)**2))
#         rmse_atp.append(np.sqrt((atp_pred - uvar)**2))
#         err_ata.append(ata_err)
#         err_atp.append(atp_err)
#         diff_ata_atp.append(ata_pred - atp_pred)
#         diff_err_ata_atp.append(ata_err - atp_err)
#
#
# df = pd.DataFrame(
#     data={
#         'rmse ata': rmse_ata,
#         'rmse atp': rmse_atp,
#         'err ata': err_ata,
#         'err atp': err_atp,
#         'ata-atp': diff_ata_atp,
#         'err ata - err atp': diff_err_ata_atp
#     }
# )
#
# df.describe().to_csv('test_ata_pk_data/compare_ata_atp_directional.csv')

smoothed = smooth_area_to_point_pk(
    semivariogram_model=variogram,
    blocks=blocks,
    point_support=point_support,
    number_of_neighbors=8,
    max_range=maximum_range
)

smoothed.to_file('test_ata_pk_data/smoothed_directional.shp')
