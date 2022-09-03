"""
Authors:

Sean Lim | @seanjunheng2
"""
import requests
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None)

def download_air_quality(dataset):
    r"""Return DataFrame of latest air quality measurements in Poland"""
    assert dataset in ['CO', 'SO2', 'PM2.5', 'PM10', 'NO2', 'O3', 'C6H6']

    api_output = requests.get('http://api.gios.gov.pl/pjp-api/rest/station/findAll').json()
    stations_df = pd.json_normalize(api_output)[['id', 'gegrLat', 'gegrLon']].rename(columns={'id': 'stationId'})

    sensors_df = []
    for i in stations_df["stationId"].values:
        api_output = requests.get('http://api.gios.gov.pl/pjp-api/rest/station/sensors/' + str(i)).json()
        sensors_df.append(pd.json_normalize(api_output))
    sensors_df = pd.concat(sensors_df, ignore_index=True)

    values_df = []
    for i in sensors_df[sensors_df['param.paramCode'] == dataset]['id'].values:
        api_output = requests.get('http://api.gios.gov.pl/pjp-api/rest/data/getData/' + str(i)).json()['values']
        try:
            values_df.append([i, pd.DataFrame.from_dict(api_output).dropna()['value'].values[0]])
        except:
            values_df.append([i, np.nan])
    values_df = pd.DataFrame(data=values_df, columns=['id', dataset])

    final_df = pd.merge(sensors_df[['id', 'stationId']], stations_df, on='stationId')
    final_df = pd.merge(final_df, values_df, on='id')
    final_df = final_df.drop(['id', 'stationId'], axis=1).rename(columns={'gegrLat': 'latitude', 'gegrLon': 'longitude'})
    return final_df

print(download_air_quality('CO'))
print(download_air_quality('SO2'))
print(download_air_quality('PM2.5'))
print(download_air_quality('PM10'))
print(download_air_quality('NO2'))
print(download_air_quality('O3'))
print(download_air_quality('C6H6'))
