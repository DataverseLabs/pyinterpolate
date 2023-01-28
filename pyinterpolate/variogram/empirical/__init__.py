"""
Experimental Variogram and Covariogram calculations.

The first step of spatial data analysis and kriging is to estimate experimental semivariance.
    A researcher calculates semivariance between point pairs at a different lags (distances).
    The created profile is used to estimate theoretical semivariogram.

Module allows to calculate and create:
- omnidirectional, directional, weighted experimental semivariogram,
- omnidirectional, directional experimental covariogram,
- an omnidirectional or a directional variogram point cloud,
- analyse statistical properties of the variogram point cloud,
- show the variogram point cloud as scatter plot per lag, box plots per lag or violin plots per lag.

We provide input points [x, y, value], step size and max range for all cases;
    additionally, direction, semi-major & semi-minor axes ratio of ellipsis to calculate directional cases;
    additionally, weights array to calculate weughted semivariance.

Changelog
---------

| Date       | Change description         | Author         |
|------------|----------------------------|----------------|
| 2022-02-16 | First release of docstring | @SimonMolinsky |

Authors
-------
- Szymon Molinski @SimonMolinsky

Bibliography
------------
[1] Armstrong, M. Basic Linear Geostatistics. Springer 1998. doi:10.1007/978-3-642-58727-6
[2] Oliver, M. and Webster, R. Basic Steps in Geostatistics: The Variogram and Kriging. Springer 2015.
    doi:10.1007/978-3-319-15865-5

"""


from pyinterpolate.variogram.empirical.cloud import build_variogram_point_cloud, VariogramCloud
from pyinterpolate.variogram.empirical.experimental_variogram import build_experimental_variogram, ExperimentalVariogram
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance
from pyinterpolate.variogram.empirical.covariance import calculate_covariance
