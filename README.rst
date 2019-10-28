PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies.

Bibliography
------------

PyInterpolate was created thanks to many resources and all of them are pointed here:

- GIS Algorithms by Ningchuan Xiao: https://uk.sagepub.com/en-gb/eur/gis-algorithms/book241284
- Pardo-Iguzquiza E., VARFIT: a fortran-77 program for fitting variogram models by weighted least squares, Computers & Geosciences 25, 251-261, 1999
- Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units, Mathematical Geology 40(1), 101-128, 2008
- Deutsch C.V., Correcting for Negative Weights in Ordinary Kriging, Computers & Geosciences Vol.22, No.7, pp. 765-773, 1996

Requirements
------------

* Python 3.7+

* Numpy 1.17.3+

* Pandas 0.25.2+

* GeoPandas 0.6.1+

* Matplotlib 3.1.1+

Package structure
-----------------

::

 - [-] Kriging implementation
        - [X] Distance calculation
        - [X] Empirical semivariogram calculation
        - [X] Theoretical semivariogram modeling
        - [X] Ordinary Kriging
        - [X] Simple Kriging
        - [ ] Regression Kriging
        - [ ] Area-to-Point interpolation
        - [-] Area-to-Area interpolation
        - [ ] Poisson Kriging

 - [X] Data visualization and interpolation
        - [X] Experimental semivariogram
        - [X] Experimental and Theoretical semivariogram
        - [X] 2D point grid
        - [X] 2D raster

 - [X] Additional scripts
        - [X] Read and prepare data
        - [X] Interpolation results as a matrix
        - [X] False administrative units development

 - [ ] Tutorials
        - [X] Distance calculation
        - [-] Ordinary Kriging
        - [-] Simple Kriging
        - [ ] Regression Kriging
        - [ ] Poisson Kriging

Bugs
====

- [-] Negative values in estimated error variance in ordinary kriging: https://github.com/szymon-datalions/pyinterpolate/issues/3

Issues
======

- [-] Complete documentation and description of "Random geographical units" class