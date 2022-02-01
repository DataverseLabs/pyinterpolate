# Areal data transformation

To disaggregate areal data into the point support, one must know a regionalized variable’s point support covariance or semivariance. Then the semivariogram deconvolution is performed. In this iterative process, the experimental semivariogram of areal data is transformed to fit the semivariogram model of a linked point support variable. Journel and Huijbregts [1] presented a general approach to deconvolute regularized semivariogram:

1. Define a point-support model from inspection of the semivariogram od areal data and estimate the parameters (sill and range) using basic deconvolution rules.
2. Compute the theoretically regularized model and compare it to the experimental curve.
3. Adjust the parameters of the point-support model to bring them in line with the regularized model.

Pyinterpolate follows an extended procedure. It leads to the automatic semivariogram regularization. [2] described this process in detail. The procedure has ten steps:

1. Compute the experimental semivariogram of areal data and fit a theoretical model to it.
2. The algorithm compares a few theoretical models and calculates the error between a modeled curve and the experimental semivariogram. The algorithm selects the theoretical model with the lowest error as the initial point-support model.
3. The initial point-support model is regularized according to the procedure given in [2].
4. Quantify the deviation between the initial point-support model and the theoretically regularized model.
5. The initial point-support model, the regularized model and the associated deviation are considered optimal at this stage.
6. Iterative process begins: for each lag, the algorithm calculates the experimental values for the new point-support semivariogram. Those values are computed through a rescaling of the optimal point support model available at this stage.
7. The rescaled values are fitted to the new theoretical model in the same procedure as the second step.
8. The new theoretical model (from step 7.) is regularized.
9. Compute the difference statistic for the new regularized model (step 8.). Decide what to do next based on the value of the new difference statistic. If it is smaller than the optimal difference statistic, use the point support model (step 7.) and the associated statistic as the optimal point-support model and the optimal difference statistic. Repeat steps from 6. to 8. If the difference statistic is larger or equal to the optimal difference statistic, repeat steps 6 through 8 with a change of the rescaling weights.
10. Stop the procedure after i-th iteration whenever one of the specified criteria are met: (1) the difference statistic reaches a sufficiently small value, (2) the maximum number of iterations has been tried, (3) a small decrease in the difference statistic was recorded a given number of times.

## Bibliography

[1] Journel AG, Huijbregts CJ. Mining geostatistics. Academic Press; London: 1978. p. 600.

[2] Goovaerts, P. (2007). Kriging and semivariogram deconvolution in the presence of irregular geographical units. Mathematical Geosciences, 40, 101–128. https://doi.org/10.1007/s11004-007-9129-1