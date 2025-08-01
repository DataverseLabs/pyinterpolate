Changes - from version >= 1.x
=============================

2025-07-17
----------

**version 1.0.3**

* [bug] `TheoreticalVariogram().__repr__()` method bug when optional `custom_weights` are not given

2025-07-16
----------

**version 1.0.2**

* [bug] neighbors selection in `Deconvolution` was falling when `Blocks` index was string, now it works with any index type

2025-07-11
----------

**version 1.0.1**

* [feature] `ordinary_kriging()` and `simple_kriging()` functions allow user to pass array-like object of coordinates as the `unknown_locations` parameter.
* [enhacement] added `__repr__()` method to `TheoreticalVariogram` class.


Changes in transition between 0.x to 1.x
========================================

The transition to the stable version of the package brings changes that break the past API and require your attention.

The most significant changes are related to `Blocks` and `PointSupport` classes. Their scope is broader, and they serve as core data structures when you want to perform Poisson Kriging of polygons. We recommend going through API tutorials describing both objects.

New classes and functions
-------------------------

* `CentroidPoissonKrigingInput`
* `ndarray_pydantic`, a type-checking function
* `ExperimentalVariogramModel`
* `RawPoints`
* `VariogramPoints`
* `PoissonKrigingInput`
* `SemivariogramErrorModel`
* `TheoreticalVariogramModel`
* `BlockPoissonKriging`
* `Deviation`
* `filter_blocks`
* `smooth_blocks` is a renamed version of the old function `smooth_area_to_point_pk`
* `interpolate_points`
* `validate_plot_attributes_for_experimental_variogram` is a renamed version of the old function `validate_plot_attributes_for_experimental_variogram_class`
* `validate_bins`
* `validate_direction_and_tolerance`
* `validate_semivariance_weights`
* `build_mask_indices`
* `calculate_angular_difference`
* `clean_mask_indices`
* `define_whitening_matrix`
* `get_triangle_edges`
* `triangle_mask`
* `select_values_in_range_from_dataframe`
* `regularize`
* `weighted_avg_point_support_semivariances`
* `mean_relative_difference`
* `symmetric_mean_relative_difference`
* `calculate_average_p2b_semivariance`
* `covariance_fn`
* `from_ellipse`
* `from_ellipse_cloud`
* `from_triangle`
* `from_triangle_cloud`
* `directional_weighted_semivariance`
* `omnidirectional_variogram_cloud`
* `omnidirectional_variogram`
* `semivariance_fn`
* `directional_covariance`
* `omnidirectional_covariance`
* `directional_semivariance_cloud`
* `omnidirectional_semivariance`
* `point_cloud_semivariance`
* `TheoreticalIndicatorVariogram` is a renamed version of the old class `IndicatorVariograms`
* `get_lags`
* `get_current_and_previous_lag`
* `TheoreticalModelFunction`
* `weight_experimental_semivariance`
* `points_to_lon_lat`
* `parse_point_support_distances_array`
* `angles_to_unknown_block`
* `block_to_blocks_angles`
* `block_base_distances`
* `set_blocks_dataset`
* `parse_kriging_input`

The list of breaking changes
----------------------------

Please update your code accordingly:

* `ClusterDetector` class has been removed due to the dependency issues, but it will be reintroduced in the next update
* `calc_point_to_point_distance` has been renamed to `point_distance`
* `calculate_angular_distance` has been renamed to `calculate_angular_difference` (because it is a difference between two angles)
* `select_values_between_lags` has been renamed to `select_values_in_range` (function covers more cases than the selection of values between the variogram lags)
* `read_txt` is removed, use `GeoPandas` or `Pandas` instead
* `read_csv` is removed, use `GeoPandas` or `Pandas` instead
* `read_block` is removed, use `GeoPandas` instead
* `WeightedBlock2BlockSemivariance` is removed
* `WeightedBlock2PointSemivariance` is removed
* `weights_array` is not available as a public function
* `KrigingObject` has been removed, but other data models have been introduced
* `ExperimentalFeatureWarning` is removed
* the `kriging` function is removed
* `BlockPK` has been renamed to `BlockPoissonKriging`
* `smooth_area_to_point_pk` has been renamed to `smooth_blocks`
* `BlockToBlockKrigingComparison` has been removed
* `block_arr_to_dict` has been removed
* `block_dataframe_to_dict` has been removed
* `get_areal_centroids_from_agg` has been removed
* `get_areal_values_from_agg` has been removed
* `point_support_to_dict` has been removed but it will be reintroduced in the upcoming updates
* `transform_ps_to_dict` has been removed
* `transform_blocks_to_numpy` has been removed
* `IndexColNotUniqueError` has been removed
* `WrongGeometryTypeError` has been removed
* `SetDifferenceWarning` has been removed
* `check_ids` has been removed
* `get_aggregated_point_support_values` has been removed
* `get_distances_within_unknown` has been removed
* `get_study_max_range` has been removed
* `prepare_pk_known_areas` has been removed
* `select_poisson_kriging_data` has been removed
* `select_neighbors_pk_centroid_with_angle` has been removed
* `select_neighbors_pk_centroid` has been removed
* `select_centroid_poisson_kriging_data` has been removed
* `omnidirectional_point_cloud` has been removed
* `directional_point_cloud` has been removed
* `build_variogram_point_cloud` has been removed
* `omnidirectional_covariogram` has been removed
* `directional_covariogram` has been removed
* `build_experimental_variogram` has been removed
* `directional_semivariogram` has been removed
* `IndicatorVariograms` renamed to `TheoreticalIndicatorVariogram`
* `inblock_semivariance` has been removed
* `MetricsTypeSelectionError` has been removed
* `VariogramModelNotSetError` has been removed
* `validate_direction` has been removed
* `validate_points` has been removed
* `validate_tolerance` has been removed
* `validate_weights` has been removed
* `validate_selected_errors` has been removed
* `check_nuggets` has been removed
* `check_ranges` has been removed
* `check_sills` has been removed
* `validate_plot_attributes_for_experimental_variogram_class` renamed to `validate_plot_attributes_for_experimental_variogram`
* `validate_theoretical_variogram`
* `to_tiff` has been removed due to the dependency problems. The functionality will be reintroduced in the future


Old changes (version < 1.x)
===========================

2025-01-04
----------
**version 0.5.4**

* (python) removed Python 3.8 from the supported versions,
* (debug) fixed `DivisionByZeroWarning` when semivariogram range is equal to 0

2024-10-26
----------
**version 0.5.3**

* (logic) debugged variance error calculations for Area-to-Point Poisson Kriging
* (bug) https://github.com/DataverseLabs/pyinterpolate/issues/428
* (enhancement) added universal kriging functionality along with multivariate regression

2024-06-26
----------
**version 0.5.2**

* (dependencies) `GeoPandas` >= 1; `numpy` < 2

2024-02-19
----------

**version 0.5.1** (*pre production development*)

* (enhancement) `interpolate_raster()` function takes `allow_approx_solutions` parameter, and it protects from `LinAlgError` that might occur if interpolation points are duplicated (due to the floating point number representation).
* (refactoring) `calc_point_to_point_distance` function refactored to `point_distance`, changed input parameters' schema,
* (refactoring) new selection method for unequally spaced bins: `select_values_between_lags`
* (debug) `np.float` type casting has been changed to `float`

2023-09-16
----------

**version 0.5.0.post1**

* (debug) `hdbscan` is removed from requirements, cluster detection algorithms are blocked, and those will be reimplemented in the closest future. The `HDBSCAN` package breaks installation of the package.

2023-08-29
----------

**version 0.5**

* (feature) `to_tiff()` function which writes kriging output from the `interpolate_raster()` function to `tiff` and `tfw` files,
* (debug) `safe` theoretical variogram models,
* (enhancement) `model_types` parameter can be string only (in the future the name of this parameter will be changed),
* (dependencies) fixed dependencies (`hdbscan` and `scikit-learn`),
* (enhancement) updated tutorials, we slightly changed their structure,
* (dependencies) End of support for Python 3.7,
* (invalid) Warning when user tries to use `.plot()` method of the `ExperimentalVariogram` class,
* (invalid) Default `direction` and `tolerance` are `None` instead of floats,
* (invalid) Removed unnecessary warning from the `.autofit()` method.

2023-05-03
----------

**version 0.4.2**

* (enhancement) added support for reading `feather` and `parquet` files by Point Support and Blocks classes.

2023-04-15
----------

**version 0.4.1**

* (change) The initialization of `ExperimentalVariogram` instance always calculates variance automatically (in the previous versions users may decide if they want to).
* (enhancement) `"safe"` method of variogram autofit that chooses *linear*, *power*, and *spherical* models,
* (enhancement) add automatic nugget selection for `TheoreticalVariogram().autofit()` method,
* (debug) `Deconvolution().fit()` and `Deconvultion().fit_transform()` transform nugget, range, and sill to `float` to avoid errors related to `numpy` `int64` casting.

2023-04-02
----------

**version 0.4**

* (feature) Cluster detection with DBSCAN,
* (feature) Cluster aggregation,
* (feature) Gridding algorithm,
* (feature) Grid aggregation,
* (feature) Removed connections to external APIs, and `requests` package from requirements,
* (feature) The new package with datasets has been created: https://pypi.org/project/pyinterpolate-datasets/2023.0.0/
* (feature) Theoretical Variogram calculates not Spatial Dependence Index,
* (debugging) `rang` key in theoretical semivariogram model renamed to `range`,
* (feature) Indicator Kriging.

2023-02-09
----------

**version 0.3.7**

* (enhancement) added logging to Poisson Kriging ATP process,
* (test) added functional test for `smooth_blocks` function,
* (debug) too broad exception in `download_air_quality_poland` is narrowed to `KeyError`,
* (enhancement) log points that cannot be assigned to any area in `PointSupport` class,
* (enhancement) `transform_ps_to_dict()` function takes custom parameters for lon, lat, value and index,
* (test) `check_limits()` function tests,
* (test) plotting function of the `VariogramCloud()` class is tested and slightly changed to return `True` if everything has worked fine,
* (tutorials) new tutorial about `ExperimentalVariogram` and `VariogramCloud` classes,
* (test) new tests for `calculate_average_semivariance()` function from `block` module,
* (enhancement) function `inblock_semivariance` has been optimized,
* (docs) updated `__init__.py` of `variogram.theoretical` module,
* (enhancement) scatter plot represented as a swarm plot in `VariogramCloud`,
* (enhancement) added directional kriging for ATA and ATP Poisson Kriging,
* (debug) warning for directional kriging functions,
* (enhancement) initialization of `KrigingObject` dataclass,
* (ci/cd) added new workflow tests for MacOS and Ubuntu,
* (enhancement) added logging to Simple Kriging process.


2023-01-16
----------

**version 0.3.6**

* (enhancement) Directional Centroid-based Poisson Kriging,
* (debug) Added origin (unknown point) to calculate directional Kriging and directional Centroid-based Poisson Kriging,
* (docs) Directional Ordinary Kriging tutorial,
* (engancement) logging of area to area PK function,
* (enhancement) `tests` package moved outside the main package,
* (feature) ordinary kriging from covariance terms,
* (feature) area-to-area PK from covariance terms,
* (debug) area-to-area PK debugged,
* (feature) area-to-point PK from covariance terms,
* (debug) area-to-point PK debugged,
* (feature) centroid-based PK from covariance terms,
* (debug) centroid-based PK debugged.


2022-11-05
----------

**version 0.3.5**

* (debug) Updated directional variogram algorithm: now angle moves counterclockwise (instead of clockwise).
* (feature) Directional Ordinary Kriging,
* (feature) Directional Simple Kriging,
* (feature) Angle calculations (angle to origin, angle between vectors),
* (enhancement) `direction` parameter is `None` default, to avoid hard-to-track bugs,
* (debug) debugged `interpolate_raster()` function,
* (enhancement) kriging data selection - a small refactoring,
* (docs) Updated `distance` module docs,
* (enhancement) point kriging functions refactoring and update, better management of a singular matrices and duplicated points.


2022-10-22
----------

**version 0.3.4.post1**

* (setup) added `pyogrio` to dependencies due to the new `fiona` version (1.8.22) and `gdal` errors.

2022-10-21
----------

**version 0.3.4**

* (debug) control of data *dtypes* after transformations and preparation of `PointSupport` and `Blocks`,
* (debug) updated data selection methods for Poisson Kriging to avoid mixing column of numerical and non-numerical values in a single numpy array, (it makes algorithm faster),
* (update) updated tutorials,
* (feature) check area and point support indexes with `smooth_area_to_point_pk()`,
* (docs) updated docstrings for `calculate_covariance()` and `calculate_semivariance()` functions.


2022-10-18
----------

**version 0.3.3**

* Semivariogram `Deconvolution` takes possible model types as a parameter,
* Semivariogram `Deconvolution` uses **basic** set of variogram models (*spherical*, *linear*, *power*, *exponential*),
* New class: `DirectionalVariogram` calculates experimental variograms in four directions along with isotropic variogram,
* Corrected directions (angles were described wrong 0 degrees is W-E, -90 deg is N-S direction),
* Directional variogram calculations are faster due to the change of selection method (only non-weighted case in this release),
* Numpy's `sqrt()` method casts ints into floats (see `Issue 306 <https://github.com/DataverseLabs/pyinterpolate/issues/306>`_),
* Users can pass a nugget for `autofit()` method of `TheoreticalVariogram()` class, the same for `Deconvolution()` process.

2022-10-08
----------

**version 0.3.2**

* new test dataset with regular blocks,
* more tests for `Deconvolution`, `area_to_point_pk()`, `area_to_area_pk()`, and `centroid_based_pk()`,
* if there are no values for a given lag in experimental variogram `RunetimeError()` is raised,
* `average_block_to_block_semivariances()` appends 0 to the lags with 0 points,
* `calculate_block_to_block_semivariance()` - valid calculation of number of point pairs.


2022-09-29
----------

**version 0.3.1**

* experimental variogram, covariogram, and variogram cloud function and classes check if there are NaN's in the input data and raise `ValueError`,
* the length of major and minor axes of a directional variogram ellipsis are calculated differently from the `tolerance` parameter, (now we have a less of chaos),
* tutorial for directional variograms (Basic),
* updated `download_air_quality_poland()` function, now it can store downloaded data,
* updated documentation.

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