![status](https://joss.theoj.org/papers/3f87f562264c4e5174d9e6ed6d8812aa/status.svg) ![License](https://img.shields.io/github/license/szymon-datalions/pyinterpolate) ![Documentation Status](https://readthedocs.org/projects/pyinterpolate/badge/?version=latest) [![CodeFactor](https://www.codefactor.io/repository/github/dataverselabs/pyinterpolate/badge)](https://www.codefactor.io/repository/github/dataverselabs/pyinterpolate)

![Pyinterpolate](https://github.com/DataverseLabs/pyinterpolate/blob/main/logov03.jpg?raw=true  "Pyinterpolate logo")

# Pyinterpolate

**version 0.3.7** - *Kyiv*

---

Pyinterpolate is the Python library for **spatial statistics**. The package provides access to spatial statistics tools used in various studies. This package helps you **interpolate spatial data** with the *Kriging* technique.

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

1. *Ordinary Kriging* and *Simple Kriging* (spatial interpolation from points),
2. *Centroid-based Poisson Kriging* of polygons (spatial interpolation from blocks and areas),
3. *Area-to-area* and *Area-to-point Poisson Kriging* of Polygons (spatial interpolation and data deconvolution from areas to points).
4. *Inverse Distance Weighting*.
5. *Semivariogram regularization and deconvolution*.
6. *Semivariogram modeling and analysis*.

## How it works

The package has multiple spatial interpolation functions. The flow of analysis is usually the same for each method:

**[1.]** Read and prepare data.

```python
from pyinterpolate import read_txt

point_data = read_txt('dem.txt')
```

**[2.]** Analyze data, calculate the experimental variogram.

```python
from pyinterpolate import build_experimental_variogram

search_radius = 500
max_range = 40000

experimental_semivariogram = build_experimental_variogram(input_array=point_data,
                                                          step_size=search_radius,
                                                          max_range=max_range)
```

**[3.]** Data transformation, fit theoretical variogram.

```python
from pyinterpolate import build_theoretical_variogram

semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram,
                                      model_type='spherical',
                                      sill=400,
                                      rang=20000,
                                      nugget=0)
```

**[4.]** Interpolation.

```python
from pyinterpolate import kriging

unknown_point = (20000, 65000)
prediction = kriging(observations=point_data,
                     theoretical_model=semivar,
                     points=[unknown_point],
                     how='ok',
                     no_neighbors=32)
```

**[5.]** Error and uncertainty analysis.

```python
print(prediction)  # [predicted, variance error, lon, lat]
```

```bash
>> [211.23, 0.89, 20000, 60000]
```

With **pyinterpolate**, we can retrieve the point support model from blocks. Example from _Tick-borne Disease Detector_ study for European Space Agency - COVID-19 population at risk mapping. We did it with the Area-to-Point Poisson Kriging technique from the package. Countries worldwide aggregate disease data to protect the privacy of infected people. But this kind of representation introduces bias to the decision-making process. To overcome this bias, you may use Poisson Kriging. Block aggregates of COVID-19 infection rate are transformed into new point support semivariogram created from population density blocks. We get the population at risk map:
![Covid-19 infection risk in Poland for 14th April, 2020.](https://github.com/DataverseLabs/pyinterpolate/blob/main/deconvoluted_risk_areas.jpg?raw=true  "Covid-19 infection risk in Poland for 14th April, 2020.")

## Status

Beta (late) version: the structure will be in most cases stable, new releases will introduce new classes and functions instead of API changes.


## Setup

Setup with *conda*: `conda install -c conda-forge pyinterpolate`

Setup with *pip*: `pip install pyinterpolate`

Detailed instructions on how to install the package are presented in the file [SETUP.md](https://github.com/DataverseLabs/pyinterpolate/blob/main/SETUP.md). We pointed out there most common problems related to third-party packages.

You may follow those setup steps to create a *conda* environment with the package for your work:

### Recommended - conda installation

[1.] Create conda environment with Python >= 3.8. Recommended is Python 3.10.

```shell
conda create -n [YOUR ENV NAME] -c conda-forge python=3.10 pyinterpolate
```

[2.] Activate environment.

```
conda activate [YOUR ENV NAME]
```

[3.] You are ready to use the package!

### pip installation

With **Python>=3.7** and system ```libspatialindex_c.so``` dependencies you may install package by simple command:

```
pip install pyinterpolate
```

A world of advice, you should **always** use Virtual Environment for the installation. You may consider using PipEnv too.

## Tests and contribution

All tests are grouped in the `test` directory. If you would like to contribute, then you won't avoid testing, but it is described step-by-step here: [CONTRIBUTION.md](https://github.com/DataverseLabs/pyinterpolate/blob/main/CONTRIBUTION.md)

## Commercial and scientific projects where library has been used

* Tick-Borne Disease Detector (Data Lions company) for the European Space Agency (2019-2020).
* B2C project related to the prediction of demand for specific flu medications (2020).
* B2G project related to the large-scale infrastructure maintenance (2020-2021).
* E-commerce service for reporting and analysis, building spatial / temporal profiles of customers (2022+).
* The external data augmentation for e-commerce services (2022+).

## Community

Join our community in Discord: [Discord Server Pyinterpolate](https://discord.gg/3EMuRkj)


## Bibliography

PyInterpolate was created thanks to many resources and all of them are pointed here:

- Armstrong M., Basic Linear Geostatistics, Springer 1998,
- GIS Algorithms by Ningchuan Xiao: https://uk.sagepub.com/en-gb/eur/gis-algorithms/book241284
- Pardo-Iguzquiza E., VARFIT: a fortran-77 program for fitting variogram models by weighted least squares, Computers & Geosciences 25, 251-261, 1999,
- Goovaerts P., Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units, Mathematical Geology 40(1), 101-128, 2008
- Deutsch C.V., Correcting for Negative Weights in Ordinary Kriging, Computers & Geosciences Vol.22, No.7, pp. 765-773, 1996


## How to cite

Moliński, S., (2022). Pyinterpolate: Spatial interpolation in Python for point measurements and aggregated datasets. Journal of Open Source Software, 7(70), 2869, https://doi.org/10.21105/joss.02869


## Requirements and dependencies (v 0.3.+)

Core requirements and dependencies are:

* Python >= 3.8 (Python 3.7 may be used with `pip` installation but it won't be supported in the future).
* descartes
* geopandas
* matplotlib
* numpy
* tqdm
* pyproj
* scipy
* shapely
* fiona
* rtree
* prettytable
* pandas
* dask
* requests

You may check a specific version of requirements in the `setup.cfg` file.

## Package structure

High level overview:

 - [x] `pyinterpolate`
    - [x] `distance` - distance calculation,
    - [x] `idw` - inverse distance weighting interpolation,
    - [x] `io` - reads and prepares input spatial datasets,
    - [x] `kriging` - Ordinary Kriging, Simple Kriging, Poisson Kriging: centroid based, area-to-area, area-to-point,
    - [x] `pipelines` - a complex functions to smooth a block data, download sample data, compare different kriging techniques, and filter blocks,
    - [x] `processing` - core data structures of the package: `Blocks` and `PointSupport`, and additional functions used for internal processes,
    - [x] `variogram` - experimental variogram, theoretical variogram, variogram point cloud, semivariogram regularization & deconvolution,
    - [x] `viz` - interpolation of smooth surfaces from points into rasters.
 - [x] `tutorials` - tutorials (Basic, Intermediate and Advanced).

## API documentation

https://pyinterpolate.readthedocs.io/en/latest/
