PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies.

Bibliography
------------

PyInterpolate was created thanks to many resources and all of them are pointed here:

- GIS Algorithms by Ningchuan Xiao: https://uk.sagepub.com/en-gb/eur/gis-algorithms/book241284

Requirements
------------

* Python 3.6+

* Numpy 1.14.0+

* Matplotlib 2.1.1+

Package structure
-----------------

::

 - [ ] Kriging implementation
        - [X] Distance calculation
        - [X] Empirical semivariogram calculation
        - [X] Theoretical semivariogram modeling
        - [X] Ordinary Kriging
        - [X] Simple Kriging
        - [ ] Regression Kriging
        - [ ] Area-to-Point interpolation
        - [ ] Area-to-Area interpolation
        - [ ] Poisson Kriging

 - [ ] Data visualization and interpolation
        - [ ] Experimental semivariogram
        - [ ] Theoretical semivariogram
        - [ ] 2D point grid
        - [ ] 2D raster

 - [ ] Tutorials
        - [X] Distance calculation
        - [ ] Ordinary Kriging
        - [ ] Simple Kriging
        - [ ] Regression Kriging
        - [ ] Poisson Kriging