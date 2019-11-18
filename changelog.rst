PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies.

Changes by date
===============

13.11.2019 6:30 p.m. CET:
-------------------------

version 0.1.9

* Development: Poisson Kriging class - centroid based PK.

* Development: Weighted Semivariance calculations.

* Development: Data Preparation for centroid based PK.

-----


13.11.2019 6:30 p.m. CET:
-------------------------

version 0.1.8.2

* Development: Poisson Kriging class (not recommended to use but is possible to perform oridnary kriging with weighted distances. Covariance / semivariance calculations still need to be included.

-----


07.11.2019 3 p.m. CET:
----------------------

version 0.1.8.1.c

* Bug: sill calculation in fit_semivariance module (TheoreticalSemivariogram.fit_semivariance())

* Development: Poisson Kriging class (still not usable)

-----

04.11.2019 1 p.m. CET:
-----------------------

version 0.1.8.1.b

* Improvement: Deviation and weight calculation algorithm corrected in RegularizeModel class

* Updated Tutorial: Semivariogram Regularization

* Updated readme

-----


01.11.2019 11 a.m. CET:
-----------------------

version 0.1.8.1.a

* Tutorial: areal semivariance regularization (version alpha)

* Bug: negative errors in fit_semivariance module corrected

----


30.10.2019 11 a.m. CET:
-----------------------

version 0.1.8.1

* **semivariance_base** module corrections (use of the **get_centroids** function)

* corrected bugs: distance array for areas centroids was calculated based on the values and ids (and it shouldn't be calculated in that way). Array indexing was introduced.

* Poisson Kriging - implementation starts

----


28.10.2019 5 p.m. CET:
----------------------

version: 0.1.8

* Function **get_centroids()** to calculate polygon's centroid moved from Semivariance class into helper_functions module.

* Updated requirements list - geopandas and pandas.

* Updated crs equality test in **_calculate_inblock_semivariance()** method.
