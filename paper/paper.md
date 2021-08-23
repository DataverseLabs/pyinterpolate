---
title: 'Pyinterpolate: Spatial Interpolation in Python for point measurements and aggregated datasets'
tags:
  - Python
  - spatial statistics
  - spatial interpolation
  - Poisson Kriging
  - Semivariogram Deconvolution
  - public health
authors:
  - name: Szymon Moliński
    orcid: 0000-0003-3525-2104
    affiliation: 1
affiliations:
  - name: Data Lions company, Poland, https://datalions.eu
    index: 1
date: 20 October 2020
bibliography: paper.bib
---

# Summary

Spatial Interpolation techniques are used to interpolate values at unknown locations or filter and smooth existing data sources. Those methods work for point observations and areal aggregates. The basic idea under this family of algorithms is that every point in space can be described as a function of its neighbors values weighted by the relative distance from the analyzed point. It is known as Tobler's First Law of Geography, which states: *everything is related to everything else, but near things are more related than distant things* [@Tobler:1970].

Kriging technique designed for mining applications exploits this statement formally and nowadays it has gained a lot of attention outside the initial area of interest. Today *Kriging* is a set of methods that can be applied to problems from multiple fields: environmental science, hydrogeology, natural resources monitoring, remote sensing, epidemiology and ecology and even computer science [@Chilès:2018]. Commonly Kriging interpolates values from point measurements or regular block units but the real-world datasets are quite often different. Especially challenging are measurements of processes over areas, as for example administrative units in every country [@Goovaerts:2007]. 

Pyinterpolate tackles the areas of irregular shapes and sizes with Area-to-Area and Area-to-Point Poisson Kriging functions. With those algorithms Pyinterpolate is useful for social, environmental and public health scientists with the areal interpolation and filtering tools. Moreover, the package offers basic point Kriging and Inverse Distance Weighting techniques. Those are utilized in every field of research where geostatistical (distance) analysis gives meaningful results. Pyinterpolate merges basic Kriging techniques with more sophisticated Area-to-Area and Area-to-Point Poisson Kriging methods.


# Statement of need

Pyinterpolate is Python package for spatial interpolation. It performs predictions from point measurements and areal aggregates of different sizes and shapes. Pyinterpolate automates Kriging interpolation and semivariogram regularization and helps with data exploration, data preprocessing and semivariogram analysis. Researchers with geostatistical background have control over the basic modeling parameters: semivariogram models, nugget, sill and range, number opf neighbors included in the interpolation and Kriging type. Thing that makes Pyinterpolate different from other spatial interpolation packages is the ability to perform Kriging on areas of different shapes and sizes and this type of operation which is important in social, medical and ecological sciences. Poisson Kriging is widely used in the social sciences, epidemiology and spatial statistics [@Goovaerts:2007; @Goovaerts:2008; @Kerry:2013].

## Importance of areal (block) Kriging

Areas of irregular shapes and sizes are especially challenging for analysis and modelling. However, there are many applications where the ability for areal data modelling is desired. A good example is the health sector, as data aggregation happens over large areas due to personal data protection, but this process introduces bias to modelling and makes policy-making more complex. The main three reasons behind the transformation of the values aggregated over administrative regions which are represented as polygons of irregular shapes and sizes into a point-support model are:

1. The presence of extremely unreliable rates that typically occur for sparsely populated areas and rare events. Small population or a small sampling effort is placed in a denominatior of a rate calculation, as example number of laukemia cases (numerator) per population size in a given county (denominator) or number of whales observed in a given area (numerator) per time of observation (denominator). In those cases extreme values of observations may be related to the fact that variance for a given area of interest is high (low nomber of samples) and not to the fact that the chance of the event is particularly high for this region. 
2. The visual bias resulting from the data aggregation over administrative units with various shapes and sizes.
3. The mismatch of spatial supports for aggregated data and other variables. Data for spatial modeling should have the same spatial scale and extent for each data source and aggregated datasets are not an exception. It may lead to the trade-off where other variables are aggregated to fit areal data but a lot of information is lost in this case. The other problem is that administrative regions are artificial constructs and aggregation of variables may remove spatial trends from a data. A downscalling of areal data into a filtered population blocks may be better suited to risk estimation along with remote-sensed data or in-situ observations of correlated variables [@Goovaerts:2006].

In this context, Area-to-Area Poisson Kriging serves as the noise-filtering algorithm or areal interpolation model, and Area-to-Point Poisson Kriging interpolates and transforms values and preserves coherence of the prediction - finally, the the disaggregated estimates sum is equal to the baseline area value [@Goovaerts:2008]. Area-to-Point Poisson Kriging can be useful in the chained-models systems where we need the change of support to perform a study. Study of this type was performed by author for the project related to the pipeline with machine learning model based on the satellites data and geostatistical population-at-risk model based on Area-to-Point Poisson Kriging (the research outcomes are not published yet).

Alternatively to the Area-to-Area and Area-to-Point Poisson Kriging, researchers  may use centroids of areas and perform point kriging over a prepared regular point grid. However, this method has pitfalls. Different sizes and shapes of units leads to the imbalanced number of variogram point pairs per lag. The centroid-based approach does not catch spatial variability of the linked variable, for example, population density over an area in the context of infection rates.

To disaggregate areal data into point support one must know point support covariance and/or semivariance of a regionalized variable. Then the semivariogram deconvolution is performed. In this iterative process experimental semivariogram of areal data is transformed to fit the semivariogram model of a linked point support variable. A general approach to deconvolute regularized semivariogram was presented by Journel and Huijbregts [@journel_huijbregts78]:

1. Define a point-support model from inspection of the semivariogram od areal data and estimate of the parameters (sill and range) using basic deconvolution rules.
2. Compute the theoretically regularized model and compare to the experimental curve.
3. Adjust the parameters of the point-support model to bring them in line with the regularized model.

Pyinterpolate follows extended procedure which leads to the automatic semivariogram regularization. Process is described in [@Goovaerts2007] and in summary it has ten steps:

1. Compute the experimental semivariogram of areal data and fit a theoretical model to it.
2. Algorithm compares few types of theoretical models and the error between a modeled curve and the experimental semivariogram is calculated. The theoretical model with lowest error is selected as the initial point-support model.
3. The initial point-support model is regularized according to the procedure given in [@Goovaerts2007].
4. Quntify the deviation between the initial point-support model and the theoretically regularized model.
5. The initial point-support model, the regularized model and the associated deviation are considered as optimal at this stage.
6. Iterative process begins: for each lag experimental values for the new point-support semivariogram are calculated. Those values are computed through a rescaling of the optimal point support model available at this stage.
7. The rescaled values are fitted to the new theoretical model in the same procedure as in the second step.
8. The new theoretical model (from the step 7.) is regularized.
9. Compute the difference statistic for the new regularized model (step 8.). Decide what to do next based on the value of the new difference statistic. If it is smaller than the optimal difference statistic at this stage use the point support model (step 7.) and the associated statistic as the optimal point-support model and the optimal difference statistic. Repeat steps from 6. to 8. If the difference statistic calculated at this point is larger or equal to the optimal difference statistic repeat steps 6 throiugh 8 with a change of the rescalling weights.
10. Stop the procedure after i-th iteration whenever one of the specified criteria are met: (1) the difference statistic reaches a sufficiently small value, (2) the maximum number of iterations has been tried, (3) a small decrease in the difference statitc was recorded a given number of times.

# Methodology

The chapter presents the general calculations methodology within a package. The concrete use case is presented in the document [here](https://github.com/szymon-datalions/pyinterpolate-paper/blob/main/paper/supplementary%20materials/example_use_case.md). Comparison of algorithm with the **gstat** package is available [here](https://github.com/szymon-datalions/pyinterpolate-paper/blob/main/paper/supplementary%20materials/comparison_to_gstat.md).

## Spatial Interpolation with Kriging

Kriging is an estimation method that gives the best unbiased linear estimates of point values or block averages [@Armstrong:1998]. It is the core method of the **Pyinterpolate** package. Kriging minimizes the variance of a dataset with missing values. The main technique is the Ordinary Kriging where the value at unknown location $\hat{z}$ is estimated as a linear combination of $K$ neighbours with the observed values $z$ and weights $\lambda$ assigned to those neighbours (1).

(1) $$\hat{z} = \sum_{i=1}^{K}\lambda_{i}z_{i}$$

Weights $\lambda$ are a solution of following system of linear equations (2):

(2) $$\sum_{j=1}\lambda_{j} C(x_{i}, x_{j}) - \mu = \bar{C}(x_{i}, V); i=1, 2, ..., K$$ $$\sum_{i}\lambda_{i} = 1$$

where $C(x_{i}, x_{j})$ is a covariance between points $x_{i}$ and $x_{j}$, $\bar{C}(x_{i}, V)$ is an average covariance between point $x_{i}$ and all other points in a group ($K$ points) and $\mu$ is a process mean. The same system may be solved with semivariance instead of covariance (3):

(3) $$\sum_{i=1}\lambda_{j} \gamma(x_{i}, x_{j}) + \mu = \bar{\gamma}(x_{i}, V); i=1, 2, ..., K$$ $$\sum_{i}\lambda_{i} = 1$$

where $\gamma(x_{i}, x_{j})$ is a semivariance between points $x_{i}$ and $x_{j}$, $\bar{\gamma}(x_{i}, V)$ is an average semivariance between point $x_{i}$ and all other points.
Semivariance is a key concept of spatial interpolation. It is a measure of a dissimilarity between observations in a function of distance. Equation (4) is an experimental semivariogram estimation formula.

(4) $$\frac{1}{2N}\sum_{i}^{N}(z_{(x_{i} + h)} - z_{x_{i}})^{2}$$

where $z_{x_{i}}$ is a value at location $x_{i}$ and $z_{(x_{i} + h)}$ is a value at a translated location in a distance $h$ from $x_{i}$.

In the next step theoretical models are fitted to the experimental curve. Pyinterpolate package implements linear, spherical, exponential and gaussian models [@Armstrong:1998]. The model with the lowest error is used in (3) to estimate $\gamma$ parameter.

Ordinary Kriging is one of the classic Kriging types within the package. **Simple Kriging** is another available method for point interpolation. Simple Kriging may be used when the process mean is known over the whole sampling area. This situation rarely occurs in the real world. It can be observed in places where sampling density is high [@Armstrong:1998]. Simple Kriging system is defined as:

(5) $$\hat{z} = R + \mu$$

where $\mu$ is a process mean and $R$ is a residual at a specific location. The residual value is derived as the first element (denoted as $\boldsymbol{1}$) from:

(6) $$R = ((Z - \mu) \times \lambda)\boldsymbol{1}$$

The number of values depends on the number of neighbours in a search radius, similar to equation (1) for Ordinary Kriging. $\lambda$ weights are the solution of the following function:

(7) $$\lambda = K^{-1}(\hat{k})$$

where $K$ is a semivariance matrix between each neighbour of size $NxN$ and $k$ is a semivariance between unknown point and known points of size $Nx1$.

The package allows the use of three main types of Poisson Kriging: Centroid-based Poisson Kriging, Area-to-Area Poisson Kriging and Area-to-Point Poisson Kriging. Each of them defines the risk over areas (or points) similarly to the equation (1). However, the weights associated with the $\lambda$ parameter are estimated with additional constraints related to the population weighting. The spatial support of each unit needs to be accounted for in both the semivariogram inference and kriging. The Poisson Kriging interpolation of areal data is presented in [@Goovaerts:2006] and semivariogram deconvolution which is an intermediate step in Poisson Kriging is described in [@Goovaerts:2007].

## Interpolation methods within Pyinterpolate

**Pyinterpolate** performs six types of spatial interpolation at the time of paper writing; five types of Kriging and inverse distance weighting:

1. **Ordinary Kriging**. It is a universal method for point interpolation.
2. **Simple Kriging** is a special case of the point interpolation when the mean of the spatial process is known and does not vary spatially in a systematic way.
3. **Centroid-based Poisson Kriging**. is used for areal interpolation and filtering and assumes that each block can collapse into its centroid. It is much faster than Area-to-Area and Area-to-Point Poisson Kriging but introduces bias related to the area's transformation into single points.
4. **Area-to-Area Poisson Kriging** is used for areal interpolation and filtering. If point support varies over an area, it will appear in the analysis. The model can catch this variation.
5. **Area-to-Point Poisson Kriging**. Areal support is deconvoluted in regards to the point support. Output map has spatial resolution of the point support while coherence of analysis is preserved (sum of rates is equal to the output of Area-to-Area Poisson Kriging). It is used for point-support interpolation and data filtering.

The user starts with semivariogram exploration and modeling. Next, researcher or algorithm chooses the theoretical model which best fits the semivariogram. If this is done automatically then algorithm tests linear, spherical and exponential models with different ranges and the constant sill against the experimental curve. Model performance is measured by the root mean squared error between the theoretical model of a given type and a range with the experimental semivariance. Areal data interpolation, especially transformation from areal aggregates into point support maps, requires deconvolution of areal semivariogram. This is an automatic process which can be performed without prior knowledge of the kriging and spatial statistics. Process is described in details in [@Goovaerts:2007]. The last step is Kriging itself.

**Pyinterpolate** allows for Poisson Kriging of the datasets which are representing the aggregated counts within the blocks. Those counts should follow a Poisson distribution. Example of it is the number of disease cases per county.

Ordinary Kriging as an interpolation technique that could be applied to the many real-world problems and works well within multiple scenarios. Technique predicts values from the point measurements.

Predicted data is stored as a DataFrame known from the *Pandas* and *GeoPandas* Python packages. Pyinterpolate allows user to transform the point data into a regular Numpy array grid for the further processing and analysis. Use case with the whole scenario is available in the [paper package repository](https://github.com/szymon-datalions/pyinterpolate-paper).

The package can automatically perform the semivariogram fitting step with a derivation of the theoretical semivariogram from the experimental curve. The semivariogram regularization is completely automated beacuse it is an iterative procedure of finding the specific optimum criteria (the procedure is described in [@Goovaerts:2007]). User can change derived theoretical model only with a direct overwritting of the derived semivariogram models parameters (nugget, sill, range, model type). 

**Pyinterpolate** was initially developed for epidemiological study, where areal aggregates of infections were transformed to point support population-at-risk maps and multiple potential applications follow this algorithm. Initial field of study (epidemiology) was the reason behind automation of the tasks related to the semivariogram modeling. It is assumed that users without a wide geostatistical background may use Pyinterpolate for spatial data modeling and analysis, especially users which are observing processes related to the human population.

The example of a process where deconvolution of areal counts and semivariogram regularization occurs is presented in the \autoref{fig1}.

## Modules

Pyinterpolate has seven modules that cover all operations needed to perform spatial interpolation: from input/output operations, data processing and transformation, semivariogram fit to Kriging interpolation. \autoref{fig2} shows package structure.

![Structure of Pyinterpolate package.\label{fig2}](fig2_modules.png)

Modules follow typical data processing and modeling steps. The first module is **io_ops** which reads point data from text files and areal or point data from shapefiles, then changes data structure for further processing. **Transform** module is responsible for all tasks related to changes in data structure during program execution. Sample tasks are:

- finding centroids of areal data,
- building masks of points within lag.

Functions for distance calculation between points and between areas (blocks) are grouped within **distance** module. **Semivariance** module is the most complex part of Pyinterpolate package. It has three special classes for the calculation and storage of different types of semivariograms (experimental, theoretical, areal and point types) and other functions important for spatial analysis:

- experimental semivariance / covariance calculation,
- weighted semivariance estimation,
- variogram cloud preparation,
- outliers removal.

**Kriging** module contains Ordinary Kriging, Simple Kriging, Centroid-based Poisson Kriging, Area-to-Area Poisson Kriging and Area-to-Point Poisson Kriging algorithms. Areal models are derived from [@Goovaerts:2008], Simple Kriging and Ordinary Kriging models are based on [@Armstrong:1998].

It is possible to show output as a **Numpy** array with **viz** module and to compare multiple Kriging models trained on the same dataset with the **misc** module. The evaluation metric for comparison is an average root mean squared error over multiple random divisions of a passed dataset.

# Comparison to Existing Software

Pyinterpolate is a one package from an ecosystem of spatial modeling and spatial interpolation packages written in Python. The main difference between Pyinterpolate and other packages is that it focuses on areal deconvolution methods and Poisson Kriging techniques useful for ecology, social science and public health studies in the presented package. Potential users may choose other packages if their study is limited to point data interpolation.

The most similar and significant package from the Python environment is **PyKrige** [@benjamin_murphy_2020_3991907]. PyKrige is designed especially for point kriging. PyKrige supports 2D and 3D ordinary and universal Kriging. User is able to incorporate his/her own semivariogram models and to use external functions (as example from **scikit-learn** package [@scikit-learn]) to model drift in universal Kriging. Package is well designed, and it is actively maintained.

**GRASS GIS** [@GRASS_GIS_software] is well-established software for vector and raster data processing and analysis. GRASS contains multiple modules and its functionalities can be accessed from multiple interfaces: GUI, command line, C API, Python APU, Jupyter Notebooks, web, QGIS and R. GRASS has two functions for spatial interpolation: `r.surf.idw` and `v.surf.idw`. Both use Inverse Distance Weighting technique, first interpolated raster files and second vectors (points).

**PySAL** is the next GIS / geospatial package that is used for interpolation – but this time at the scale of blocks with assumption that each each block is treated as a node in the network. Package’s **tobler** module can be used to interpolate areal values of specific variable at different scales and sizes of support [@eli_knaap_2020_4385980]. Moreover, package has functions for multisource regression, where raster data is used as auxiliary information to enhance interpolation results. Conceptually tobler package is close to the Pyinterpolate, where main algorithm transforms areal data into point support derived from auxiliary variable.

**R programming language** offers **gstat** package for spatial interpolation and spatial modeling [@PEBESMA2004683]. Package is designed for variogram modelling, simple, ordinary and universal point or block kriging (with drift), spatio-temporal kriging and sequential Gaussian (co)simulation. Gstat is a solid package for Kriging and spatial interpolation and has the largest number of methods to perform spatial modelling. The main difference between gstat and Pyinterpolate is availability of area-to-point Poisson Kriging based on the algorithm proposed by Goovaerts [@Goovaerts:2007] in Pyinterpolate package. Comparison to **gstat** is available in the [paper repository](https://github.com/szymon-datalions/pyinterpolate-paper).

# Appendix\label{appendix}

1. [**Paper repository** with additional materials](https://github.com/szymon-datalions/pyinterpolate-paper)
2. [**Package repository**](https://github.com/szymon-datalions/pyinterpolate)
3. [**Automatic fit of semivariogram within the package**]()
4. [**Outliers Detection within the package**]()

# References

