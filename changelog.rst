PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies.

Changes by date
===============

30.10.2019 11 a.m. CET:

version 0.1.8.1

* **semivariance_base** module corrections (use of the **get_centroids** function)

* corrected bugs: distance array for areas centroids was calculated based on the values and ids (and it shouldn't be calculated in that way). Array indexing was introduced.

* Poisson Kriging - implementation starts

28.10.2019 5 p.m. CET:

version: 0.1.8

* Function **get_centroids()** to calculate polygon's centroid moved from Semivariance class into helper_functions module.

* Updated requirements list - geopandas and pandas.

* Updated crs equality test in **_calculate_inblock_semivariance()** method.
