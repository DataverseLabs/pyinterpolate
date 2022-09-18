Pyinterpolate
=============

Pyinterpolate is the Python library for **geostatistics**. The package provides access to spatial statistics tools used in various studies. This package helps you **interpolate spatial data** with the *Kriging* technique.

Changes by date
===============

2022-XX-XX
----------

**version 0.3.1**

* experimental variogram, covariogram, and variogram cloud function and classes check if there are NaN's in the input data and raise `ValueError`,
* the length of major and minor axes of a directional variogram ellipsis are calculated differently from the `tolerance` parameter, (now we have a less of chaos),

2022-09-04
----------

**version 0.3.0**

* module `io_ops` renamed to `io`,
* the refactored function `read_point_data` (old) into `read_txt`, new functions to read csv and blocks data,
* the new objects to store block data and its point-support: `Blocks` and `PointSupport`,
* Kriging is now supported by **functions**, not by classes, to speed up some calculations. In the future, classes will be introduced again,
* user has much more control over the variograms development. `ExperimentalVariogram` class calculates *variance*, *covariance*, and *semivariance*, has own plotting function. `TheoreticalVariogram` has more models to fit, and gives more control to search for the best fit - the algorithm searches over ranges and sills. Nugget is still fixed,
* module `pipelines` has the function for the block data smoothing (area-to-point Poisson Kriging), the class for block data filtering (area-to-area Poisson Kriging), the kriging comparison class, and method to download sample air pollution data,
* there are many small changes and API transformations... The package is faster and more stable,
* it works with Python 3.7, 3.8, 3.9, and 3.10,
* Ordinary and Simple Kriging of large datasets may be performed in parallel,
* the package has a few warnings and raises custom errors,
* `setup.py` is removed, now package installs from `setup.cfg`,
* data structures are more complex, but they allow user to be more flexible with an input.


2021-12-31
----------

**version 0.2.5**

* neighbors selection (lags counting) has been changed,
* `TheoreticalSemivariogram` searches for optimal sill in a grid search algorithm,
* corrected error in `Krige` class; now calculation of error variance is correct.

2021-12-11
----------

**version 0.2.4**

* `self.points_values` chenged to `self.points_array` in `TheoreticalSemivariogram` class,
* `NaN` values are tested and checked in `calc_semivariance_from_pt_cloud()` function,
* new semivariogram models included in the package: **cubic**, **circular**, **power**,
* corrected calculation of the closest neighbors for kriging interpolation,
* changed `prepare_kriging_data()` function,
* the new optional parameter `check_coordinates` (**bool**) of `calc_point_to_point_distance()` function to control the coordinates uniqueness tests. This test is very resource-consuming and should be avoided in a normal work and it should be performed before data injection into the modeling pipeline.
* the new `dev/profiling/` directory to test and profile parts of a code.

2021-08-23
----------

**version 0.2.3.post1**

* the outliers removal function: you can choose side for outlier detection and remove. Default is top, available are: both, top, down,
* the outliers removal function: changed algorithm,
* new tutorial about outliers and their influence on the final model.

2021-05-13
----------

**version 0.2.3**

* more parameters to store (and access) in TheoreticalSemivariogram class,
* error weighting against the linear regression model (ax + b),
* global mean for Simple Kriging as a required parameter,
* tqdm progress bar to `RegularizedSemivariogram.transform()` and `interpolate_raster()` functions,
* refactored Semivariogram Regularization: ranges are controlled by algorithm, not an user,
* added pull request template,
* added issues templates,
* bug in spherical semivariogram model,
* experimental variogram as points (not a solid line),
* inverse distance weighting function: algorithm, tests, documentation and new tutorial,
* changed output names of regularized data (`ArealKriging.regularize_data`) from **estimated value** to **reg.est** and from **estimated prediction error** to **reg.err**,
* error related to the id column as a string removed,
* TheoreticalSemivariogram `params` attribute changed to `nugget`, `sill` and `range` attributes.

2021-03-10
----------

**version 0.2.2.post2**

* directional semivariograms methods, docs and tests added,
* check if points are within elliptical area around point of interest method, docs and tests added,
* broken dependency in `README.md` corrected.

2021-03-02
----------

**version 0.2.2.post1**

* variogram point cloud methods, tutorials, docs and tests added,
* updated tutorials and baseline datasets to show examples with spatial correlation,
* updated `README.md`: contribution, example, sample image,
* data is tested against duplicates (points with the same coordinates),
* removed bug in `interpolate_raster()` method.
