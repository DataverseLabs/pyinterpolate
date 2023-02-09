"""
Theoretical Semivariogram implementation.

Long description
----------------
Theoretical semivariogram is a model of experimental semivariance. This model is described by a set of functions
    known as conditional negative semi-definite functions [1]. In practice, we cannot simply use any GLM model to
    describe variogram because it could lead to negative variances at some distances.
The basic properties of theoretical semivariogram [2] are:
    - Increasing variance with increasing distance or, in other words, increasing dissimilarity with distance between
      points.
    - Upper bound, named sill. If semivariogram is second-order stationary then sill is equal to data variance.
      Otherwise, it is a variance level where the change of variance is relatively small between lags. Some variograms
      are unbounded and this indicates that the process is not second-order stationary.
    - Range. It is a distance where spatial dependence exists. In practice, it is estimated as 0.95 of a distance
      where the variogram reaches sill.
    - Nugget. Theoretically semivariance at distance 0 should be equal to zero. But some processes that are not
      spatially correlated at small distances may introduce non-zero initial variance. Measurement technique may
      introduce additional bias, which is also treated as the nugget.
    - Anisotropy. Spatial variation may be different in a different directions. Consider temperature - its variogram
      function will be different in a South-North axis than West-East axis.

Researcher must model variogram from experimental observations to perform spatial interpolation. The variogram itself
    brings information about the modeled data. This module contains class and methods to fit experimental variances
    into theoretical functions. It can be done manully or automatically.

Module contains
---------------

#############
# functions #
#############

- build_theoretical_variogram() : Function is a wrapper into ``TheoreticalVariogram`` class and its ``fit()`` method.
- circular_model() : Function calculates circular model of semivariogram.
- cubic_model() : Function calculates cubic model of semivariogram.
- exponential_model() : Function calculates exponential model of semivariogram.
- gaussian_model() : Function calculates gaussian model of semivariogram.
- linear_model() : Function calculates linear model of semivariogram.
- power_model() : Function calculates power model of semivariogram.
- spherical_model() : Function calculates spherical model of semivariogram.

#############
#  classes  #
#############

- TheoreticalVariogram : Theoretical model of a spatial dissimilarity.

Flow of analysis
----------------
Experimental Variogram -> Theoretical Variogram -> Kriging
......................    ---------------------    .......

The Theoretical Variogram is created upon the Experimental Variogram. Array of lags and semivariances are passed into
    the Theoretical Semivariogram class. The class can calculate experimental semivariance automatically but
    researcher lost a part of a control over data - lag size and max range are set by algorithm.

Theoretical variogram is a specific mathematical model fitted into a data with three parameters that control the
    model shape. From the algorithmic perspective there are four unknowns:
    - model type: it is chosen from a finite number of model types,
    - sill: the parameter that describes semivariance when spatial dependency is not relevant, the maximum dissimilarity
      between pairs. Usually it is set as a dataset variance. Not all variograms have a sill,
    - range: a distance at which variogram reaches approximately 0.95 of sill, the range of spatial dependency,
    - nugget: bias at a zero distance. It is usually set to zero and automatic grid search uses 0. nugget if
      researcher doesn't force algorithm to check different value(s).

Module returns the Theoretical Variogram as a set of four values: model type (str), sill (float), range (float),
    nugget (float). It is used later for Kriging.

Variogram Models:
- circular [4], cubic, exponential, gaussian, linear, power, spherical [3].


Changelog
---------

| Date       | Change description           | Author         |
|------------|------------------------------|----------------|
| 2022-02-16 | First release of v0.3 module | @SimonMolinsky |
| 2023-01-28 | Added description of functions and classes | @SimonMolinsky |

Authors
-------
(1) Scott Gallacher | @scottgallacher-3
(2) Szymon Moliński | @SimonMolinsky

Contributors
------------
(1) Ethem Turgut | @ethmtrgt

References
----------
* variogram.empirical : Experimental Variogram and Covariogram calculations.
* variogram.regularization : Block variograms and semivariogram regularization.

Bibliography
------------
[1] Webster, R., Oliver, M.A. Geostatistics for environmental scientist (2nd ed.). ISBN: 978-0-470-02858-2. Wiley 2007.
[2] Oliver, M.A., Webster, R. Basic steps in geostatistics: the variogram and Kriging. ISBN: 978-3-319-15865-5.
    Springer 2015.
[3] Armstrong, M. Basic Linear Geostatistics. ISBN: 978-3-642-58727-6. Springer 1998.
[4] McBratney, A. B., Webster R. Choosing Functions for Semivariograms of Soil Properties and Fitting Them
    to Sampling Estimates. Journal of Soil Science 37: 617–639. 1986.
"""