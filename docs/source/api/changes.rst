Changes between version 0.x and 1.x
===================================

The new release of ``pyinterpolate`` has introduced API changes that can break old pipelines. This document covers those changes.
The most profound decision was to remove I/O adapters. ``Pyinterpolate`` doesn't handle file reading, and user should use other packages to read spatial data (for example ``GeoPandas`` or ``Pandas``).
Not all of those functions, classes, and models are listed in the official user-facing API documentation!

(*This document is not completed, and some changes will be updated in the closest future*)

Functions
---------

.. list-table:: Functions
   :widths: 50 50
   :header-rows: 1

   * - version 0.x
     - version 1.x
   * - ``calc_point_to_point_distance()``
     - changed to ``point_distance()``
   * - ``calculate_angular_distance()``
     - changed to ``calculate_angular_difference()``
   * - ``select_values_between_lags()``
     - changed to ``select_values_in_range()``
   * - ``weights_array()``
     - now private ``_weights_array()``
   * - ``smooth_area_to_point_pk()``
     - changed to ``smooth_blocks()``
   * - ``validate_plot_attributes_for_experimental_variogram_class()``
     - changed to ``validate_plot_attributes_for_experimental_variogram()``

Classes
-------

.. list-table:: Classes
   :widths: 50 50
   :header-rows: 1

   * - version 0.x
     - version 1.x
   * - ``BlockPK()``
     - changed to ``BlockPoissonKriging()``
   * - ``Blocks()``
     - heavily refactored and optimized
   * - ``PointSupport()``
     - heavily refactored and optimized
   * - ``IndicatorVariograms()``
     - changed to ``TheoreticalIndicatorVariogram()``

Temporarily not available functions and classes
-----------------------------------------------

- ``ClusterDetector()`` (due to the dependency issues)
- ``point_support_to_dict()``

New functions and classes
-------------------------

- ``CentroidPoissonKrigingInput()`` - class model for centroid-based Poisson Kriging operation
- ``ndarray_pydantic`` - annotation model
- ``ExperimentalVariogramModel`` - data model of Experimental Variogram
- ``RawPoints`` - data model of raw points
- ``VariogramPoints`` - data model of variogram points
- ``SemivariogramErrorModel`` - data model of semivariogram error types
- ``TheoreticalVariogramModel`` - data model of theoretical variogram
- ``filter_blocks()`` - filters block aggregates using Poisson Kriging
- ``smooth_blocks()`` - transforms aggregated data into point support model
- ``BlockPoissonKriging()`` - Area-to-Area, Area-to-Point, or Centroid-based Poisson Kriging regularization
- ``interpolate_points()`` - ordinary kriging of multiple points
- ``interpolate_points_dask()`` - same a/a but using dask, for large datasets
- ``validate_plot_attributes_for_experimental_variogram()``
- ``validate_bins()``
- ``validate_direction_and_tolerance()``
- ``validate_semivariance_weights()``
- ``build_mask_indices()``
- ``calculate_angular_difference()``
- ``clean_mask_indices()``
- ``define_whitening_matrix()``
- ``get_triangle_edges()``
- ``triangle_mask()``
- ``select_values_in_range_from_dataframe()``
- ``regularize()`` - semivariogram regularization / deconvolution
- ``weighted_avg_point_support_semivariances()``
- ``mean_relative_difference()``
- ``symmetric_mean_relative_difference()``
- ``Deviation()`` - class handles deviation monitoring during semivariogram regularization
- ``calculate_average_p2b_semivariance()``
- ``covariance_fn()`` - calculate covariance
- ``from_ellipse()`` - directional variogram method
- ``from_ellipse_cloud()`` - a/a but for variogram cloud
- ``from_triangle()`` - directional variogram method
- ``from_triangle_cloud()`` - a/a but for variogram cloud
- ``directional_weighted_semivariance()``
- ``omnidirectional_variogram_cloud()``
- ``omnidirectional_variogram()``
- ``semivariance_fn()``
- ``directional_covariance()``
- ``omnidirectional_covariance()``
- ``directional_semivariance_cloud()``
- ``omnidirectional_semivariance()``
- ``point_cloud_semivariance()``
- ``TheoreticalIndicatorVariogram()`` - Indicator Variogram
- ``get_lags()``
- ``get_current_and_previous_lag()``
- ``TheoreticalModelFunction()`` - theoretical functions for variogram modeling
- ``weight_experimental_semivariance()``
- ``points_to_lon_lat()``
- ``parse_point_support_distances_array()``
- ``angles_to_unknown_block()``
- ``block_to_blocks_angles()``
- ``block_base_distances()``
- ``set_blocks_dataset()``
- ``parse_kriging_input()``

Functions and classes that are no longer supported
--------------------------------------------------

- ``to_tiff()``
- ``read_txt()``
- ``read_csv()``
- ``read_block()``
- ``WeightedBlock2BlockSemivariance()``
- ``WeightedBlock2PointSemivariance()``
- ``KrigingObject()``
- ``ExperimentalFeatureWarning()``
- ``kriging()`` - instead use ``ordinary_kriging()`` or ``simple_kriging()``
- ``BlockToBlockKrigingComparison()``
- ``block_arr_to_dict()``
- ``block_dataframe_to_dict()``
- ``get_areal_centroids_from_agg()``
- ``get_areal_values_from_agg()``
- ``transform_ps_to_dict()``
- ``transform_blocks_to_numpy()``
- ``IndexColNotUniqueError()``
- ``WrongGeometryTypeError()``
- ``SetDifferenceWarning``
- ``check_ids()``
- ``get_aggregated_point_support_values()``
- ``get_distances_within_unknown()``
- ``get_study_max_range()``
- ``prepare_pk_known_areas()``
- ``select_poisson_kriging_data()``
- ``select_neighbors_pk_centroid_with_angle()``
- ``select_neighbors_pk_centroid()``
- ``select_centroid_poisson_kriging_data()``
- ``omnidirectional_point_cloud()``
- ``directional_point_cloud()``
- ``build_variogram_point_cloud()``
- ``omnidirectional_covariogram()``
- ``directional_covariogram()``
- ``directional_semivariogram()``
- ``inblock_semivariance()``
- ``MetricsTypeSelectionError()``
- ``VariogramModelNotSetError()``
- ``validate_direction()``
- ``validate_points()``
- ``validate_tolerance()``
- ``validate_weights()``
- ``validate_selected_errors()``
- ``check_nuggets()``
- ``check_ranges()``
- ``check_sills()``
- ``validate_theoretical_variogram()``
- ``to_tiff()`` - might be returned, but for now it is removed due to the dependency issues




