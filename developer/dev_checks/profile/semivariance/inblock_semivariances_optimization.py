import numpy as np


def some_model(x):
    return (100 + 9.8*x)


def inblock_semivariance(distances_between_points) -> float:
    semivariances = some_model(distances_between_points)

    average_block_semivariance = np.sum(semivariances) / len(semivariances)
    return average_block_semivariance


def inblock_semivariance_unique(distances_between_points) -> float:

    # TODO: part below to test with very large datasets
    unique_distances, uniq_count = np.unique(distances_between_points, return_counts=True)  # Array is flattened here
    semivariances = some_model(unique_distances)
    multiplied_semivariances = semivariances * uniq_count

    average_block_semivariance = np.sum(multiplied_semivariances) / len(semivariances)
    return average_block_semivariance


if __name__ == '__main__':
    import pandas as pd
    from datetime import datetime

    outputs = []

    for idx in range(0, 20):
        for arr_size in np.logspace(1, 7, 20):
            for x_limit in [10, 100, 1000, 5000, 20000, 100000]:
                arr_size = int(arr_size)
                x_limit = int(x_limit)

                print(idx, arr_size, x_limit)

                arr = np.random.randint(0, x_limit, arr_size)

                t_start_0 = datetime.now()
                _ = inblock_semivariance(arr)
                secs_0 = (datetime.now() - t_start_0).total_seconds()

                t_start_1 = datetime.now()
                _ = inblock_semivariance_unique(arr)
                secs_1 = (datetime.now() - t_start_1).total_seconds()

                d = {
                    'idx': idx,
                    'arr_size': arr_size,
                    'max_element': x_limit,
                    'plain time': secs_0,
                    'uniq time': secs_1,
                    'experiment': 'integer random numbers'
                }

                outputs.append(d)

    df = pd.DataFrame(outputs)
    df.to_csv('inblock_times_integer.csv')
