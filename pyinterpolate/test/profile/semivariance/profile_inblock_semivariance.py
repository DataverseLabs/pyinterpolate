# from tqdm import tqdm
# from datetime import datetime
# import numpy as np
# import matplotlib.pyplot as plt
#
# from pyinterpolate.variogram.regularization.block.inblock_semivariance import calculate_inblock_semivariance
# from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram
#
#
# SILL_VAL = 200
# RANGE_VAL = 100
# STD_VAL = np.sqrt(SILL_VAL)
# MODEL_TYPE = 'spherical'
# FAKE_THEORETICAL_MODEL = TheoreticalVariogram()
# FAKE_THEORETICAL_MODEL.from_dict({
#     'name': MODEL_TYPE,
#     'nugget': 0,
#     'sill': SILL_VAL,
#     'range': RANGE_VAL
# })
# NUMBER_OF_TRIALS = 5
# ARRAY_SIZES = np.linspace(10, 5000, 10, dtype=int)
# IDS = np.linspace(1000, 2000, 50)
#
#
# if __name__ == '__main__':
#     results_one_core = []
#     results_multiple_cores = []
#     for arr_size in tqdm(ARRAY_SIZES):
#         print(f'Fn has started analysis of an array of size {arr_size}')
#         print(datetime.now())
#         one_core = []
#         multiple_cores = []
#         for _ in tqdm(range(0, NUMBER_OF_TRIALS)):
#             test_dict = {}
#             for _id in IDS:
#                 arr_x = np.arange(0, arr_size)
#                 arr_y = np.random.randint(0, RANGE_VAL, arr_size)
#                 random_array = np.random.normal(SILL_VAL, STD_VAL, arr_size)
#                 test_matrix = np.zeros(shape=(arr_size, 3))
#                 test_matrix[:, 0] = arr_x
#                 test_matrix[:, 1] = arr_y
#                 test_matrix[:, 2] = random_array
#                 test_dict[_id] = test_matrix
#
#             t0 = datetime.now()
#             _ = calculate_inblock_semivariance(test_dict, FAKE_THEORETICAL_MODEL)
#             tend = datetime.now() - t0
#             one_core.append(tend.microseconds)
#
#             t0 = datetime.now()
#             _ = calculate_inblock_semivariance(test_dict, FAKE_THEORETICAL_MODEL)
#             tend = datetime.now() - t0
#             multiple_cores.append(tend.microseconds)
#
#         results_one_core.append(np.mean(one_core))
#         results_multiple_cores.append(np.mean(multiple_cores))
#
#     # Plot
#     plt.figure()
#     plt.plot(ARRAY_SIZES, results_one_core)
#     plt.plot(ARRAY_SIZES, results_multiple_cores)
#     plt.legend(['one core', 'multiple cores x 6'])
#     plt.show()
