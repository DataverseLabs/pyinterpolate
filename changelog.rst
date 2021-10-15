PyInterpolate
=============

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies.

Changes by date
===============

2022-XX-XX

**version 0.3.0**

* module `io_ops` renamed to `io`,
* the refactored function `read_point_data` (old) into `read_txt` (new): now it sets crs and transforms input data into a GeoDataFrame,
* the new function `read_csv` to read spatial csv files,

2021-XX-XX
----------

**version 0.2.3.XX**

* the outliers removal function: you can choose side for outlier detection and remove. Default is top, available are: both, top, down,

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
