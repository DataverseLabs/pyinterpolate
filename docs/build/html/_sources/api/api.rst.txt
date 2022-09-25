API
===

Distance calculations
---------------------

.. autofunction:: pyinterpolate.calc_point_to_point_distance
   :noindex:

.. autofunction:: pyinterpolate.calc_block_to_block_distance
   :noindex:

.....

Inverse Distance Weighting
--------------------------

.. autofunction:: pyinterpolate.inverse_distance_weighting
   :noindex:

.....

Input / Output
--------------

.. autofunction:: pyinterpolate.read_block
   :noindex:

.. autofunction:: pyinterpolate.read_csv
   :noindex:

.. autofunction:: pyinterpolate.read_txt
   :noindex:

.....

Kriging
-------

Point Kriging
.............

.. autofunction:: pyinterpolate.kriging
   :noindex:

.. autofunction:: pyinterpolate.ordinary_kriging
   :noindex:

.. autofunction:: pyinterpolate.simple_kriging
   :noindex:

Block - Poisson Kriging
.......................

.. autofunction:: pyinterpolate.centroid_poisson_kriging
   :noindex:

.. autofunction:: pyinterpolate.area_to_area_pk
   :noindex:

.. autofunction:: pyinterpolate.area_to_point_pk
   :noindex:

.....

Pipelines
---------

Kriging-based processes
.......................

.. autoclass:: pyinterpolate.BlockFilter
   :noindex:

.. autoclass:: pyinterpolate.pipelines.block_filtering.BlockPK
   :noindex:

.. autoclass:: pyinterpolate.pipelines.BlockToBlockKrigingComparison
   :noindex:

.. autofunction:: pyinterpolate.smooth_blocks
   :noindex:

Data download
.............

.. autofunction:: pyinterpolate.download_air_quality_poland
   :noindex:

Core data structures
--------------------

.. autoclass:: pyinterpolate.Blocks
   :noindex:

.. autoclass:: pyinterpolate.PointSupport
   :noindex:

Variogram
---------

Experimental
............

.. autofunction:: pyinterpolate.build_experimental_variogram
   :noindex:

.. autofunction:: pyinterpolate.build_variogram_point_cloud
   :noindex:

.. autoclass:: pyinterpolate.ExperimentalVariogram
   :noindex:

.. autoclass:: pyinterpolate.VariogramCloud
   :noindex:

Theoretical
...........

.. autofunction:: pyinterpolate.build_theoretical_variogram
   :noindex:

.. autoclass:: pyinterpolate.TheoreticalVariogram
   :noindex:

Block
.....

.. autoclass:: pyinterpolate.AggregatedVariogram
   :noindex:

Deconvolution
.............

.. autoclass:: pyinterpolate.Deconvolution
   :noindex:

Visualization
-------------

.. autofunction:: pyinterpolate.interpolate_raster
   :noindex:
