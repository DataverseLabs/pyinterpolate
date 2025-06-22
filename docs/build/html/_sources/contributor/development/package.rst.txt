Package structure
=================

High level overview:

- [x] ``src/pyinterpolate``
    - [x] ``core`` - ``Blocks`` and ``PointSupport`` classes, complex kriging pipelines,
    - [x] ``distance`` - distance calculation,
    - [x] ``evaluate`` - model evaluation tools,
    - [x] ``idw`` - inverse distance weighting interpolation,
    - [x] ``kriging`` - Ordinary Kriging, Simple Kriging, Poisson Kriging: centroid based, area-to-area, area-to-point,
    - [x] ``semivariogram`` - experimental variogram, theoretical variogram, variogram point cloud, semivariogram regularization & deconvolution,
    - [x] ``transform`` - internal transformations and data processing functions,
    - [x] ``viz`` - interpolation of smooth surfaces from points.
- [x] ``tutorials`` - tutorials (Basic, Intermediate and Advanced) and API overview.
- [x] ``tests`` - unit tests for all modules.