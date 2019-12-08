import numpy as np


def calculate_prediction_error(test_data,
                               test_data_index_name, test_data_value_name,
                               predictions, error_type='rmse'):
    errors_list = []
    for p in predictions:
        p_id = p[0]
        p_val = p[1]
        t_val = test_data[test_data[test_data_index_name] == p_id][test_data_value_name].values[0]

        if error_type == 'mse':
            err_val = (p_val - t_val)**2
        elif error_type == 'rmse':
            err_val = np.sqrt((p_val - t_val)**2)
        else:
            raise TypeError('Available error types: mse and rmse.')
        errors_list.append(err_val)

    error = np.sum(errors_list)/len(errors_list)
    return error