import pandas as pd
import numpy as np


if __name__ == '__main__':

    nn = 3

    data = [
        [0, 1, 2, 8, 6],
        [1, 0, 3, 7, 8],
        [2, 3, 0, 6, 7],
        [8, 7, 6, 0, 2],
        [6, 8, 7, 2, 0]
    ]

    df = pd.DataFrame(index=['a', 'b', 'c', 'd', 'e'], columns=['a', 'b', 'c', 'd', 'e'], data=data)
    melted = df.reset_index(names='_df_index_').melt(id_vars='_df_index_')
    df_sorted = melted.sort_values(by=['_df_index_', 'value'])
    df_sorted = df_sorted[df_sorted['value'] > 0]
    grouped = df_sorted.groupby('_df_index_').head(nn)
    ds = grouped.groupby('_df_index_')['variable'].apply(list)
    dict_ds = ds.to_dict()

    print(grouped)
    print(ds)
    print(dict_ds)
