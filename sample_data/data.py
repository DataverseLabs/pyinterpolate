import os
import sample_data


class SampleData:

    def __init__(self):
        self.module_path = os.path.dirname(sample_data.__file__)
        self.dem = os.path.join(self.module_path, 'point_data/csv/pl_dem.csv')
        self.accidents = os.path.join(self.module_path, 'areal_data/cancer_data.shp')
        self.population = os.path.join(self.module_path, 'population_data/cancer_population_base.shp')
