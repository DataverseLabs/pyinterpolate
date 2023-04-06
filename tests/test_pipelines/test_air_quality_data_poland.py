import unittest
import pandas as pd
# from pyinterpolate.pipelines.samples import download_air_quality_poland


# TODO:
# Deprecated - test should be moved to pyinterpolate-datasets package
# https://pypi.org/project/pyinterpolate-datasets/2023.0.0/

# class TestDownloadAirQualityDataPoland(unittest.TestCase):
#
#     def test_download_air_quality_data_poland(self):
#         for sample in ['CO', 'SO2', 'PM2.5', 'PM10', 'NO2', 'O3', 'C6H6']:
#             df = download_air_quality_poland(sample)
#             self.assertIsInstance(df, pd.DataFrame)
#             self.assertEqual(
#                 {'station_id', 'x', 'y', sample},
#                 set(df.columns)
#             )
#             self.assertGreater(df[sample].mean(), 0)
