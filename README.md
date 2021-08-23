![status](https://joss.theoj.org/papers/3f87f562264c4e5174d9e6ed6d8812aa/status.svg) ![License](https://img.shields.io/github/license/szymon-datalions/pyinterpolate) ![Build Status](https://travis-ci.com/szymon-datalions/pyinterpolate.svg?branch=master) ![Documentation Status](https://readthedocs.org/projects/pyinterpolate/badge/?version=latest) [![CodeFactor](https://www.codefactor.io/repository/github/szymon-datalions/pyinterpolate/badge)](https://www.codefactor.io/repository/github/szymon-datalions/pyinterpolate)

![Pyinterpolate](https://github.com/szymon-datalions/pyinterpolate/blob/main/logo.png?raw=true  "Pyinterpolate logo")

**version 0.2.3** - *Jezero Crater*
---------------------------------------

PyInterpolate is designed as the Python library for geostatistics. It's role is to provide access to spatial statistics tools used in a wide range of studies. This package helps you **interpolate spatial data** with *Kriging* technique. In the close future you'll use more spatial interpolation tools.

If you’re:

- GIS expert,
- geologist,
- mining engineer,
- ecologist,
- public health specialist,
- data scientist.

Then this package may be useful for you. You could use it for:

- spatial interpolation and spatial prediction,
- alone or with machine learning libraries,
- for point and areal datasets.

Pyinterpolate allows you to perform:

1. Ordinary Kriging and Simple Kriging (spatial interpolation from points),
2. Centroid-based Kriging of Polygons (spatial interpolation from blocks and areas),
3. Area-to-area and Area-to-point Poisson Kriging of Polygons (spatial interpolation and data deconvolution from areas to points).

How it works
--------------

Package allows o perform multiple spatial interpolation tasks. The flow of analysis is usually the same for each interpolation method:

**[1.]** Read and prepare data.

```python
from pyinterpolate.io_ops import read_point_data

point_data = read_point_data('xyz_txt_file.txt')
```

**[2.]** Analyze data, Semivariance calculation.

```python
from pyinterpolate.semivariance import calculate_semivariance

search_radius = 0.01
max_range = 0.32

experimental_semivariogram = calculate_semivariance(
	data=point_data,
	step_size=search_radius,
	max_range=max_range)
```

**[3.]** Data transformation, theoretical semivariogram.

```python
from pyinterpolate.semivariance impJezero Craterort TheoreticalSemivariogram
semivar = TheoreticalSemivariogram(points_array=point_data, empirical_semivariance=experimental_semivariogram)
number_of_ranges = 32

semivar.find_optimal_model(weighted=False, number_of_ranges=number_of_ranges)
```

**[4.]** Interpolation.

```python
from pyinterpolate.kriging import Krige

model = Krige(semivariogram_model=semivar, known_points=point_data)
unknown_point = (12.1, -5.9)

ok_pred = model.ordinary_kriging(unknown_location=unknown_point, number_of_neighbours=32)
```

**[5.]** Error and uncertainty analysis.

```python
real_val = 10  # Some real, known observation at a given point
squared_error = (real_val - ok_pred[0])**2
print(squared_error)
```

```bash
>> 48.72
```

With **pyinterpolate** you are able to retrieve point support model from areal aggregates. Example from _Tick-borne Disease Detector_ study for European Space Agency - COVID-19 population at risk mapping. It was done with Area-to-Point Poisson Kriging technique from package. Countries along the world presents infections as areal sums to protect privacy of infected people. But this kind of representaion introduces bias to the decision-making process. To overcome this bias you may use Poisson Kriging. Areal aggregates of COVID-19 infection rate are transformed to new point support semivariogram created from population density blocks. As output we get population at risk map:

![Covid-19 infection risk in Poland for 14th April, 2020.](https://github.com/szymon-datalions/pyinterpolate/blob/main/deconvoluted_risk_areas.jpg?raw=true  "Covid-19 infection risk in Poland for 14th April, 2020.")



Status
------

Beta version: package is tested and the main structure is preserved but future changes are very likely to occur.


Setup
-----

Setup by pip: pip install pyinterpolate / **Python 3.7** is required!

Detailed instructions how to setup package are presented in the file [SETUP.md](https://github.com/szymon-datalions/pyinterpolate/blob/master/SETUP.md). We pointed there most common problems related to third-party packages.

You may follow those setup steps to create conda environment with package for your tests:

### Recommended - conda installation

[1.] First install system dependencies to use package (```libspatialindex_c.so```):

LINUX:

```
sudo apt install libspatialindex-dev
```

MAC OS:

```
brew install spatialindex
```

[2.] Next step is to create conda enviornment with Python 3.7, pip and notebook packages and activate your environment:

```
conda create -n [YOUR NAME] -c conda-forge python=3.7 pip notebook
```

```
conda activate [YOUR NAME]
```

[3.] In the next step install **pyinterpolate** and its dependencies with ```pip```:

```
pip install pyinterpolate
```

[4.] You are ready to use the package!

### pip installation

With **Python==3.7** and system ```libspatialindex_c.so``` dependencies you may install package by simple command:

```
pip install pyinterpolate
```

A world of advice is to use Virtual Environment for the installation.

Tests and contribution
------------------------

All tests are grouped in `test` directory. To run them you must have installed `unittest` package. More about test and contribution is here: [CONTRIBUTION.md](https://github.com/szymon-datalions/pyinterpolate/blob/master/CONTRIBUTION.md)




Commercial and scientific projects where library has been used
--------------------------------------------------------------

* Tick-Borne Disease Detector (Data Lions company) for the European Space Agency (2019-2020).
* B2C project related to the prediction of demand for specific flu medications,
* B2G project related to the large-scale infrastructure maintenance.

Community
---------

Join our community in Discord: [Discord Server PyInterpolate](https://discord.gg/3EMuRkj)


Bibliography
------------

PyInterpolate was created thanks to many resources and all of them are pointed here:

- Armstrong M., Basic Linear Geostatistics, Springer 1998,
- GIS Algorithms by Ningchuan Xiao: https://uk.sagepub.com/en-gb/eur/gis-algorithms/book241284
- Pardo-Iguzquiza E., VARFIT: a fortran-77 program for fitting variogram models by weighted least squares, Computers & Geosciences 25, 251-261, 1999,
- Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units, Mathematical Geology 40(1), 101-128, 2008
- Deutsch C.V., Correcting for Negative Weights in Ordinary Kriging, Computers & Geosciences Vol.22, No.7, pp. 765-773, 1996

Requirements and dependencies
-----------------------------

* Python 3.7.6

* Numpy 1.18.3

* Scipy 1.4.1

* GeoPandas 0.7.0

* Fiona 1.18.13.post1 (Mac OS) / Fiona 1.8 (Linux)

* Rtree 0.9.4 (Mac OS), Rtree >= 0.8 & < 0.9 (Linux)

* Descartes 1.1.0

* Pyproj 2.6.0

* Shapely 1.7.0

* Matplotlib 3.2.1

Package structure
-----------------

High level overview:

 - [ ] pyinterpolate
    - [x] **distance** - distance calculation,
    - [x] **idw** - inverse distance weighting interpolation,
    - [x] **io_ops** - reads and prepares input spatial datasets,
    - [x] **transform** - transforms spatial datasets,
    - [x] **viz** - interpolation of smooth surfaces from points into rasters,
    - [x] **kriging** - Ordinary Kriging, Simple Kriging, Poisson Kriging: centroid based, area-to-area, area-to-point,
    - [x] **misc** - compare different kriging techniques,
    - [x] **semivariance** - calculate semivariance, fit semivariograms and regularize semivariogram,
    - [x] **tutorials** - tutorials (Basic, Intermediate and Advanced)

Functions documentation
-----------------------

Pyinterpolate https://pyinterpolate.readthedocs.io/en/latest/


Development
------------------

- multi core processing (speed up calculations),
- code refactoring to be more close to the **GeoPandas** package style,
- set to work with the newer versions of Python and Windows OS.


Known Bugs
-----------------

- Package may crash with very large dataset (memory issues). Operations are performed with numpy arrays and for datasets larger than 10 000 points there could be a memory issue ([Issue page](https://github.com/szymon-datalions/pyinterpolate/issues/64))
