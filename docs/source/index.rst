.. pyinterpolate documentation master file, created by
   sphinx-quickstart on Tue Apr  8 20:53:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pyinterpolate
=============

**version 1.0.0**
-----------------

.. note::
   The last documentation update: *2025-06-20*

``Pyinterpolate`` is the Python library for **spatial statistics**. The package provides access to spatial statistics tools (variogram analysis, Kriging, Poisson Kriging, Indicator Kriging, Inverse Distance Weighting).

If you’re:

- GIS expert
- Geologist
- Social scientist

Then this package may be useful for you. You could use it for:

- spatial interpolation and spatial prediction
- alone or with machine learning libraries
- for point observations interpolation
- and aggregated data disaggregation

You can run:

1. **Ordinary Kriging** and **Simple Kriging** - spatial interpolation from points
2. **Centroid-based Poisson Kriging** of polygons - spatial interpolation from blocks and regions
3. **Area-to-area** and **Area-to-point Poisson Kriging** of Polygons - spatial interpolation and data deconvolution from areas to points
4. **Indicator Kriging** - kriging based on probabilities
5. **Universal Kriging** - kriging with trend
6. **Inverse Distance Weighting** - benchmarking spatial interpolation technique
7. **Semivariogram regularization and deconvolution** - transforming variogram of areal data in regards to point support data
8. **Semivariogram modeling and analysis** - is your data spatially correlated? How do neighbors influence each other?

With ``Pyinterpolate`` you can transform data aggregated on a county-level to better resolution.
The example is COVID-19 population at risk mapping. Countries worldwide aggregate disease data to protect the privacy of infected people. But this kind of representation introduces bias to the decision-making process. To overcome this bias, you may use Poisson Kriging. Block aggregates of COVID-19 infection rate are transformed into the point support created from population density blocks. We get the population at risk map:

.. image:: imgs/deconvoluted_risk_areas.jpg
  :width: 400
  :alt: Covid-19 infection risk in Poland for 14th April, 2020.

Contents
--------

.. toctree::
   :maxdepth: 1

   setup/setup
   usage/quickstart
   usage/tutorials
   usage/learning_materials
   science/citation
   science/bibliography
   community/community
   api/api
   api/changes
   contributor/development

Citation
--------
Moliński, S., (2022). Pyinterpolate: Spatial interpolation in Python for point measurements and aggregated datasets. Journal of Open Source Software, 7(70), 2869, https://doi.org/10.21105/joss.02869