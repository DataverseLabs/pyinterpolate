from pyinterpolate import Blocks
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib as mpl
from pyinterpolate.distance.gridding import create_grid, points_to_grid
mpl.use('TkAgg')  # !IMPORTANT


DATASET = '../../samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 8

AREAL_INPUT = Blocks()
AREAL_INPUT.from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)

centroids = AREAL_INPUT.data[[AREAL_INPUT.cx, AREAL_INPUT.cy, AREAL_INPUT.value_column_name]]
centroids['geometry'] = gpd.points_from_xy(centroids[AREAL_INPUT.cx], centroids[AREAL_INPUT.cy])
centroids = gpd.GeoDataFrame(centroids, crs=AREAL_INPUT.data.crs)
centroids.set_geometry('geometry')

centroids.plot(column=AREAL_INPUT.value_column_name, legend=True)
plt.show()

grid = create_grid(
    centroids['geometry'], min_number_of_cells=20, grid_type='hex'
)

aggs = points_to_grid(centroids, grid)

aggs.plot(column=AREAL_INPUT.value_column_name, legend=True)
plt.show()
