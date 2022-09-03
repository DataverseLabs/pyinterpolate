# The Theory of Kriging

Kriging is an estimation method that gives the best unbiased linear estimates of point values or block averages [1]. It is the core method of the Pyinterpolate package.
The primary technique is the Ordinary Kriging. The value at unknown location $\hat{z}$ is estimated as a linear combination of $K$ neighbors with the observed values $z$ and weights $\lambda$ assigned to those neighbors (1).

(1) $$\hat{z} = \sum_{i=1}^{K}\lambda_{i}z_{i}$$

Weights $\lambda$ are a solution of following system of linear equations (2):

(2) $$\sum_{j=1}^{K}\lambda_{j} C(x_{i}, x_{j}) - \mu = \bar{C}(x_{i}, V); i=1, 2, ..., K$$ $$\sum_{i}^{K}\lambda_{i} = 1$$

where $C(x_{i}, x_{j})$ is a covariance between points $x_{i}$ and $x_{j}$, $\bar{C}(x_{i}, V)$ is an average covariance between point $x_{i}$ and all other points in a group ($K$ points) and $\mu$ is the Lagrange multiplier. The same system may be solved with semivariance instead of covariance (3):

(3) $$\sum_{i=1}^{K}\lambda_{j} \gamma(x_{i}, x_{j}) + \mu = \bar{\gamma}(x_{i}, V); i=1, 2, ..., K$$ $$\sum_{i}^{K}\lambda_{i} = 1$$

where $\gamma(x_{i}, x_{j})$ is a semivariance between points $x_{i}$ and $x_{j}$, $\bar{\gamma}(x_{i}, V)$ is an average semivariance between point $x_{i}$ and all other points.
Semivariance is a key concept of spatial interpolation. It is a measure of dissimilarity between observations in a function of distance. Equation (4) is an experimental semivariogram estimation formula and $\gamma_{h}$ is an experimental semivariance at lag $h$:

(4) $$\gamma_{h} = \frac{1}{2N}\sum_{i}^{N}(z_{(x_{i} + h)} - z_{x_{i}})^{2}$$

where $z_{x_{i}}$ is a value at location $x_{i}$ and $z_{(x_{i} + h)}$ is a value at a translated location in a distance $h$ from $x_{i}$.

Pyinterpolate package implements linear, spherical, exponential and Gaussian models [1]. They are fitted to the experimental curve. The model with the lowest error is used in (3) to estimate the $\gamma$ parameter.

**Simple Kriging** is another method for point interpolation in Pyinterpolate. We may use Simple Kriging when we know the process mean. This situation rarely occurs in real-world scenarios. It is observed in places where sampling density is high [1]. Simple Kriging system is defined as:

(5) $$\hat{z} = R + \mu$$

where $\mu$ is a Lagrange multiplier and $R$ is a residual at a specific location. The residual value is derived as the first element (denoted as $\boldsymbol{1}$) from:

(6) $$R = ((Z - \mu) \times \lambda)\boldsymbol{1}$$

The number of values depends on the number of neighbors in a search radius, similar to equation (1) for Ordinary Kriging. The weights $\lambda$ are the solution of the following function:

(7) $$\lambda = K^{-1}(\hat{k})$$

The $K$ denotes a semivariance matrix between each neighbor of size $NxN$. The $k$ parameter is a semivariance between unknown (interpolated) location and known points of size $Nx1$.

Users may use three types of Poisson Kriging procedure: Centroid-based Poisson Kriging, Area-to-Area Poisson Kriging and Area-to-Point Poisson Kriging. Each defines the risk over areas (or points) similarly to the equation (1). However, the algorithm estimates the weights associated with the $\lambda$ parameter with additional constraints related to the population weighting. The spatial support of each unit needs to be accounted for in both the semivariogram inference and kriging. The procedure of Poisson Kriging interpolation of areal data is presented in [2] and semivariogram deconvolution in [3].

## Bibliography

[1] Armstrong, M. (1998). Basic Linear Geostatistics. Springer. https://doi.org/10.1007/
266978-3-642-58727-6

[2] Goovaerts, P. (2006). Geostatistical analysis of disease data: Accounting for spatial support and population density in the isopleth mapping of cancer mortality risk using area-to-point poisson kriging. International Journal of Health Geographics, 5. https://doi.org/10.1186/1476-072X-5-52

[3] Goovaerts, P. (2007). Kriging and semivariogram deconvolution in the presence of irregular geographical units. Mathematical Geosciences, 40, 101–128. https://doi.org/10.1007/s11004-007-9129-1