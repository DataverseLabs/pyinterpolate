import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from pyinterpolate import Blocks
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl

import hdbscan
mpl.use('TkAgg')  # !IMPORTANT


DATASET = '../../samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 8

AREAL_INPUT = Blocks()
AREAL_INPUT.from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)

centroids = AREAL_INPUT.data[[AREAL_INPUT.cx, AREAL_INPUT.cy]].values

# plot
xss = centroids[:, 0]
yss = centroids[:, 1]


# Detect clusters and show those
clusterer = hdbscan.HDBSCAN(

)

clusterer.fit(centroids)

lbls = clusterer.labels_
no_of_labels = len(np.unique(lbls))

color = cm.get_cmap('Wistia')
clist = [color(i) for i in range(color.N)]
cmap = LinearSegmentedColormap.from_list('Clusters cmap', clist, color.N)
bounds = np.linspace(0, no_of_labels, no_of_labels+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1)
scat = ax.scatter(xss, yss, c=lbls, cmap=cmap, norm=norm, edgecolor='black')
cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
cb.set_label('Labels')
plt.show()



