![License](https://img.shields.io/github/license/szymon-datalions/pyinterpolate) ![Build Status](https://travis-ci.com/szymon-datalions/pyinterpolate.svg?branch=master) ![Documentation Status](https://readthedocs.org/projects/pyinterpolate/badge/?version=latest) [![CodeFactor](https://www.codefactor.io/repository/github/szymon-datalions/pyinterpolate/badge)](https://www.codefactor.io/repository/github/szymon-datalions/pyinterpolate)

PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies. This package helps you **interpolate spatial data** with *Kriging* technique. In the close future you'll use more spatial interpolation tools.

If you’re:

- GIS expert,
- geologist,
- mining engineer,
- ecologist,
- public health specialist,
- data scientist.

Then this package may be useful for you. You could use it for:

- spatial interpolation and spatial prediction,
- alone or with machine learning libraries,
- for point and areal datasets.

Pyinterpolate allows you to perform:

1. Ordinary Kriging and Simple Kriging (spatial interpolation from points),
2. Centroid-based Kriging of Polygons (spatial interpolation from blocks and areas),
3. Area-to-area and Area-to-point Poisson Kriging of Polygons (spatial interpolation and data deconvolution from areas to points).


Status
------

Beta version: package is tested and the main structure is preserved but future changes are very likely to occur.


Setup
-----

Setup by pip: pip install pyinterpolate / **Python 3.7** is required!

Manual setup is described in the file SETUP.md: https://github.com/szymon-datalions/pyinterpolate/blob/master/SETUP.md We pointed there most common problems related to third-party packages.



Commercial and scientific projects where library has been used
--------------------------------------------------------------

* Tick-Borne Disease Detector (Data Lions company) for the European Space Agency (2019-2020).
* B2C project related to the prediction of demand for specific flu medications,
* B2G project related to the large-scale infrastructure maintenance.

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
    - [x] **distance** - distance calculation
    - [x] **io_ops** - reads and prepares input spatial datasets,
    - [x] **transform** - transforms spatial datasets,
    - [x] **viz** - interpolation of smooth surfaces from points into rasters,
    - [x] **kriging** - Ordinary Kriging, Simple Kriging, Poisson Kriging: centroid based, area-to-area, area-to-point,
    - [x] **misc** - compare different kriging techniques,
    - [x] **semivariance** - calculate semivariance, fit semivariograms and regularize semivariogram,
    - [x] **tutorials** - tutorials (Basic, Intermediate and Advanced)

Functions documentation
-----------------------

Pyinterpolate https://pyinterpolate.readthedocs.io/en/latest/


Development
===========

- inverse distance weighting,
- semivariogram analysis and visualization methods,
- see Projects page of this repository!


Known Bugs
==========

- (still) not detected!
