Package structure
=================

High level overview:

- [x] ``pyinterpolate``
    - [x] ``distance`` - distance calculation,
    - [x] ``idw`` - inverse distance weighting interpolation,
    - [x] ``io`` - reads and prepares input spatial datasets,
    - [x] ``kriging`` - Ordinary Kriging, Simple Kriging, Poisson Kriging: centroid based, area-to-area, area-to-point,
    - [x] ``pipelines`` - a complex functions to smooth a block data, download sample data, compare different kriging techniques, and filter blocks,
    - [x] ``processing`` - core data structures of the package: ``Blocks`` and ``PointSupport``, and additional functions used for internal processes,
    - [x] ``variogram`` - experimental variogram, theoretical variogram, variogram point cloud, semivariogram regularization & deconvolution,
    - [x] ``viz`` - interpolation of smooth surfaces from points into rasters.
- [x] ``tutorials`` - tutorials (Basic, Intermediate and Advanced).
