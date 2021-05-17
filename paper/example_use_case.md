# Example Use Case – Breast Cancer Rates in Northeastern United States

This appendix to the paper presents sample pipeline of analysis. It is presented how Area-to-Point Poisson Kriging transforms areal counts of breast cancer cases in the Northeastern part of United States.

## Dataset

Breast cancer rates are taken from the *Incidence Rate Report for U.S.* counties and were clipped to the counties of the Northeastern part of U.S. [1]. Observations are age-adjusted and multiplied by 100,000 for the period 2013-2017.

Population centroids are retrieved from the *U.S. Census Blocks 2010* [2]. Breast cancer affects only females but for this example the whole population for an area was included. Raw and transformed datasets are available in a [dedicated Github repository](https://github.com/szymon-datalions/pyinterpolate-paper/tree/main/paper-examples/example-use-case).

Presented work is Area-to-Point Poisson Kriging of Breast Cancer areal aggregates dataset and transformation of those areal aggregates into population-specific blocks (points). This process requires two main steps: **semivariogram regularization** and **Poisson Kriging**.

## 1. Read and prepare data

The initial step of analysis is data preparation. Pyinterpolate transforms passed shapefiles to numpy arrays containing geometries and values for processing. Areal data is transformed into its `id, geometry, centroid x, centroid y` and `value`; point data is transformed into `points` and their `values` within the area with specific `id`.

## 2. Analyze and test semivariance of points or areas

Experimental semivariogram model of data must be retrieved at the beginning of analysis. Semivariance of areal centroids (Figure 1) and semivariance of point support (Figure 2) should be checked to be sure that process is spatially correlated, especially at the scale of point support. User selects the maximum range of analysis - *study extent* - and step size for each lag. The package calculates experimental semivariance based on the provided input.

![Experimental semivariogram of areal centroids.\label{fig2}](fig2.png)

*Figure 1: Experimental semivariogram of areal centroids.*

![Experimental semivariogram of point support.\label{fig3}](fig3.png)

*Figure 2: Experimental semivariogram of point support.*

Semivariogram of areal rates shows weak spatial autocorrelation but this may be the effect of data aggregation and large differences in blocks shape and size. The semivariogram of point support presents better spatial autocorrelation pattern and it reaches sill at a distance of 100 kilometers.

## 3. Create theoretical semivariogram or regularize areal aggregated values

Semivariogram modeling is fully automated and best model is selected based on the lowest error between chosen model type from *spherical*, *linear*, *gaussian* or *exponential* models and the experimental curve.

Deconvolution of areal semivariogram is a more complex problem and it's algorithm is described in [3].  Pyinterpolate implementation divides semivariogram regularization into two parts. First part is an initial preparation of data and a development of the first optimized theoretical model. In a second step a areal semivariogram is regularized in a loop. It is a time consuming process. Computation time directly depends on the number of points of the support.

Experimental semivariogram and theoretical model of areal data along with first output of regularization may be checked before the main loop to be sure that process can be modeled with Kriging method. Figure 3 presents initial (baseline) semivariograms and Figure 4 shows those after regularization. After the procedure we are able to export model for the Poisson Kriging interpolation.

![Semivariograms after fit procedure.\label{fig4}](fig4.png)

*Figure 3: Semivariograms after fit procedure.*

![Semivariograms after regularization.\label{fig5}](fig5.png)

*Figure 4: Semivariograms after complete regularization.*

Regularized Semivariogram has much more visible spatial component and sill is reached at 100 kilometers instead of half of this value. This model may be used for Poisson Krigin interpolation.

## 4. Build Kriging model and export output

With theoretical semivariogram we are able to model data with Kriging. Poisson Kriging model is used to estimate population at risk from areal aggregates. Area-to-Point Poisson Kriging requires us to know the semivariogram model and to assign the number of closest neighbors and maximum radius of neighborhood search.

Whole process may take a while, especially if there are many support points. Method `regularize_data()` returns *GeoDataFrame* object with `[id, geometry, estimated value, estimated prediction error, rmse]` columns. It may be plotted with *matplotlib* and as a result **population at risk** map is generated (Figure 5). Finally, point support map may be saved as a shapefile.

![Breast cancer population at risk map in Northeastern United States state.\label{fig6}](fig6.png)

*Figure 5: Breast cancer population at risk map in Northeastern United States state.*

Comparison of input and output data in this example is presented in Figure 6. Output values and error of variance may be used later for reporting and / or as the elements of larger modeling infrastructure.

 ![Report output.\label{fig7}](fig1_example.png)
 
 *Figure 6: Sample report output.*
 
 ---
 
# Bibliography
 
**1) Cancer Data**
 
```
{,
  author = {National Cancer Institute},
  title = {Incidence Rates Table: Breast Cancer: Pennsylvania State},
  year = 2021,
  url = {https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=42&areatype=county&cancer=055&race=00&sex=2&age=001&stage=999&year=0&type=incd&sortVariableName=rate&sortOrder=default&output=0#results},
  urldate = {2021-02-03}
}
```

**2) Census Data**

```
{,
  author = {United States Census Bureau},
  title = {Centers of Population for the 2010 Census},
  year = 2021,
  url = {https://www.census.gov/geographies/reference-files/2010/geo/2010-centers-population.html},
  urldate = {2021-02-03}
}
```

**3) Goovaerts, P. Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units**

```
{Goovaerts:2007,
  author = {Goovaerts, Pierre},
  month = {12},
  pages = {101-128},
  title = {Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units},
  doi = {10.1007/s11004-007-9129-1},
  urldate = {2020-10-24},
  volume = {40},
  year = {2007},
  journal = {Mathematical Geosciences}
}
```
 