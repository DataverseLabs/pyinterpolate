"""
The module contains functions that use public APIs to get data samples for experiments.

Authors
-------
1. Sean Lim | @seanjunheng2

"""

import requests
import numpy as np
import pandas as pd


def download_air_quality_poland(dataset: str, export=False, export_path='air_quality_sample.csv') -> pd.DataFrame:
    """
    Function downloads air quality data from Polish (Central Europe) stations. (EPSG:4326)

    Parameters
    ----------
    dataset : str
        The observed compound from the following list:
            - 'CO': carbon monoxide,
            - 'SO2': sulfur dioxide,
            - 'PM2.5':  particulate matter with size <= 2.5 micrometers,
            - 'PM10': particulate matter with size <= 10 micrometers,
            - 'NO2': nitrogen dioxide,
            - '03': ozone,
            - 'C6H6': benzene.

    export : bool, default = False
        Export loaded dataset into a csv file.

    export_path : str, default = 'air_quality_sample.csv'

    Returns
    -------
    final_df : DataFrame
        columns = ``[station_id, x, y, reading name]``

    Authors
    -------
    Sean Lim | @seanjunheng2

    TODO
    ----
    - remove too broad exception
    """
    assert dataset in ['CO', 'SO2', 'PM2.5', 'PM10', 'NO2', 'O3', 'C6H6']

    # urls
    stations_list_url = 'https://api.gios.gov.pl/pjp-api/rest/station/findAll'
    sensors_list_url = 'https://api.gios.gov.pl/pjp-api/rest/station/sensors/'
    data_url = 'https://api.gios.gov.pl/pjp-api/rest/data/getData/'

    # Cols
    lat_col = 'gegrLat'
    lon_col = 'gegrLon'
    station_id_col = 'stationId'

    api_output = requests.get(stations_list_url).json()
    stations_df = pd.json_normalize(api_output)[['id', lat_col, lon_col]].rename(columns={'id': station_id_col})

    sensors_df = []
    for i in stations_df[station_id_col].values:
        api_output = requests.get(sensors_list_url + str(i)).json()
        sensors_df.append(pd.json_normalize(api_output))
    sensors_df = pd.concat(sensors_df, ignore_index=True)

    values_df = []
    for i in sensors_df[sensors_df['param.paramCode'] == dataset]['id'].values:
        api_output = requests.get(data_url + str(i)).json()['values']
        try:
            values_df.append([i, pd.DataFrame.from_dict(api_output).dropna()['value'].values[0]])
        except KeyError:
            values_df.append([i, np.nan])
    values_df = pd.DataFrame(data=values_df, columns=['id', dataset])

    final_df = pd.merge(sensors_df[['id', station_id_col]], stations_df, on=station_id_col)
    final_df = pd.merge(final_df, values_df, on='id')
    final_df = final_df.drop(['id'], axis=1).rename(
        columns={lat_col: 'y', lon_col: 'x', station_id_col: 'station_id'}
    )

    if export:
        final_df.to_csv(export_path, index='station_id')

    return final_df
