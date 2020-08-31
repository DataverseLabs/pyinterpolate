import os
import sample_data


class SampleData:

    def __init__(self):
        self.module_path = os.path.dirname(sample_data.__file__)
        self.dem = os.path.join(self.module_path, 'poland_dem_gorzow_wielkopolski')
        self.accidents = os.path.join(self.module_path, 'areal_data/road_accidents.shp')
        self.population = os.path.join(self.module_path, 'population_data/population_centroids_poland.shp')
