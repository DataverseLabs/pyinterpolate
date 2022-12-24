import numpy as np


weights = 'weights.csv'
k = 'k.csv'


weights_arr = np.loadtxt(weights, delimiter=',', dtype=float)
k_arr = np.loadtxt(k, delimiter=',', dtype=float)

while True:
    try:
        x = np.linalg.solve(weights_arr, k_arr)
    except np.linalg.LinAlgError as _:
        weights_arr = weights_arr[:-2, :-2]

        p_ones = np.ones((weights_arr.shape[0], 1))
        predicted_with_ones_col = np.c_[weights_arr, p_ones]
        p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
        p_ones_row[0][-1] = 0.
        weights_arr = np.r_[predicted_with_ones_col, p_ones_row]

        k_arr = k_arr[:-2]
        k_ones = np.ones(1)[0]
        k_arr = np.r_[k_arr, k_ones]
    else:
        break

zhat = dataset[:, -2].dot(output_weights[:-1])

sigma = np.matmul(output_weights.T, k)

print(np.linalg.matrix_rank(weights_arr) != weights_arr.shape[0])
