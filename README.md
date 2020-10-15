PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies.


Status
------

Pre-Beta version: package is tested and the main structure is preserved but future changes are very likely to occur. Look into projects and issues tab to learn more.


Setup
-----

Setup is described in the file SETUP.md: https://github.com/szymon-datalions/pyinterpolate/blob/master/SETUP.md

Commercial and scientific projects where library has been used
--------------------------------------------------------------

* Tick-Borne Disease Detector (Data Lions company) for the European Space Agency (2019-2020).

Community
---------

Join our community in Discord: https://discord.gg/3EMuRkj


Bibliography
------------

PyInterpolate was created thanks to many resources and all of them are pointed here:

- Armstrong M., Basic Linear Geostatistics, Springer 1998,
- GIS Algorithms by Ningchuan Xiao: https://uk.sagepub.com/en-gb/eur/gis-algorithms/book241284
- Pardo-Iguzquiza E., VARFIT: a fortran-77 program for fitting variogram models by weighted least squares, Computers & Geosciences 25, 251-261, 1999,
- Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units, Mathematical Geology 40(1), 101-128, 2008
- Deutsch C.V., Correcting for Negative Weights in Ordinary Kriging, Computers & Geosciences Vol.22, No.7, pp. 765-773, 1996

Requirements and dependencies
-----------------------------

* Python 3.7.6

* Numpy 1.18.3

* Scipy 1.4.1

* GeoPandas 0.7.0

* Fiona 1.18.13.post1 (Mac OS) / Fiona 1.8 (Linux)

* Rtree 0.9.4 (Mac OS), Rtree >= 0.8 & < 0.9 (Linux)

* Descartes 1.1.0

* Pyproj 2.6.0

* Shapely 1.7.0

* Matplotlib 3.2.1

Package structure
-----------------

High level overview:

::

 - [ ] pyinterpolate
    - [x] **calculations** - distance calculation
    - [x] **data_processing** - preparation of spatial data and data processing tasks,
    - [x] **data visualization** - interpolation of smooth surfaces as rasters,
    - [x] **kriging** - Ordinary Kriging, Simple Kriging, Poisson Kriging: centroid based, area-to-area, area-to-point,
    - [x] **misc** - compare different kriging techniques,
    - [x] **semivariance** - calculate semivariance, fit semivariograms and regularize semivariogram,
    - [x] **tutorials** - tutorials (Basic, Intermediate and Advanced)

Functions documentation
-----------------------
Pyinterpolate [https://pyinterpolate.readthedocs.io/en/latest/]

Development
===========

- inverse distance weighting,
- point cloud variograms,
- semivariogram params management,
- semivariogram regularization with epidemiological data tutorial

Known Bugs
==========

- (still) not detected!
