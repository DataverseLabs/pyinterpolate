# Docstrings within Pyinterpolate

`This part of package is far from complete! We need your help with docstrings and documentation management!`

## Contents

- [Introduction](#introduction)
- [Changelog](#changelog)
- [Function Docstring](#functions)
- [Class Docstring](#classes)
- [Module Docstring](#modules)
- [More Resources and Bibliography](#resources)

## Introduction

Docstrings in **Pyinterpolate** follow the style guide of [numpy](https://numpydoc.readthedocs.io/en/latest/format.html). Our idea is to be consistent with the most important numerical packages.

## Changelog

| Date       | Change description        | Author         |
|------------|---------------------------|----------------|
| 2022-02-16 | First release of document | @SimonMolinsky |

## Functions

Parts:

1. (Required) `Short description`,
2. (Optional) Function `summary`,
3. (Required) `Parameters` (if any is passed into a function),
4. (Required) `Returns` (if function returns any value) or `Yields` (if function is a generator),
5. (Optional) `Receives` parameters send to generator with `.send()` method.
6. (Optional) `Other Parameters`.
7. (Required) `Raises`.
8. (Required) `Warns`.
9. (Optional) `Warnings` - optional warnings.
10. (Optional) `See Also` - links to the similar or important functions, classes, etc.
11. (Optional) `Notes` - equations and explanations.
12. (Required) `References` - if code is based on a specific publication or a work of someone else then it should be cited here.
13. (Required) `Examples` - example of usage.

### Template

```python

def do_something(a: float, b, c_opt=None) -> int:
    """Short description of the process.
    
    Here we write extended summary if needed. Bibliography links and usages in Notes part.
    
    Parameters
    -----------
    a : float
        Description of parameter a.
        
    b
        Description of parameter b without any type.
        
    c_opt : float or None, optional, default=None
            Description of a c_opt parameter.
            
    Returns
    -------
    output : int
             Description of output value(s)
             
    Raises
    ------
    PolygonNotValidException
        If the passed polygon is not valid.
        
    Warns
    -----
    AngularCoordinateSystemWarning
        Warning invoked if input data has geographic projection (angular coordinates)
        
    References
    ----------
    .. [1] Numpydoc maintaners. https://numpydoc.readthedocs.io/en/latest/format.html 2019
    
    Examples
    --------
    >>> import numpy as np
    >>> a_var = 1
    >>> b_var = np.arange(0, 1000)
    >>> value = do_something(a_var, b_var)
    >>> print(value)
    0
    
    """
    
    # Here are warnings and exceptions
    
    output = 0
    
    return output

```

### Example

```python
def build_experimental_variogram(input_array,
                                 step_size: float,
                                 max_range: float,
                                 weights=None,
                                 direction=0,
                                 tolerance=1) -> EmpiricalVariogram:
    """
    Function prepares:
        - experimental semivariogram,
        - experimental covariogram,
        - variance.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value) or (Point(), value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    weights : numpy array or None, optional, default=None
              weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted.

    direction : float (in range [0, 360]), optional, default=0
                direction of semivariogram, values from 0 to 360 degrees:
                * 0 or 180: is NS direction,
                * 90 or 270 is EW direction,
                * 45 or 225 is NE-SW direction,
                * 135 or 315 is NW-SE direction.

    tolerance : float (in range [0, 1]), optional, default=1
                If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                for 0 tolerance.
                * The minor axis size is (tolerance * step_size)
                * The major axis size is ((1 - tolerance) * step_size)
                * The baseline point is at a center of the ellipse.
                Tolerance == 1 creates an omnidirectional semivariogram.

    Returns
    -------
    semivariogram_stats : EmpiricalSemivariogram
        The class with empirical semivariogram, empirical covariogram and variance

    See Also
    --------
    calculate_covariance : function to calculate experimental covariance and variance of a given set of points.
    calculate_semivariance : function to calculate experimental semivariance from a given set of points.
    EmpiricalSemivariogram : class that calculates and stores experimental semivariance, covariance and variance.

    Notes
    -----
    Function is an alias for EmpiricalSemivariogram class and it forces calculations of all spatial statistics from a
        given dataset.

    Examples
    --------
    >>> import numpy as np
    >>> REFERENCE_INPUT = np.array([
    ...    [0, 0, 8],
    ...    [1, 0, 6],
    ...    [2, 0, 4],
    ...    [3, 0, 3],
    ...    [4, 0, 6],
    ...    [5, 0, 5],
    ...    [6, 0, 7],
    ...    [7, 0, 2],
    ...    [8, 0, 8],
    ...    [9, 0, 9],
    ...    [10, 0, 5],
    ...    [11, 0, 6],
    ...    [12, 0, 3]
    ...    ])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = build_experimental_variogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+--------------------+
    | lag |    semivariance    |      covariance     |    var_cov_diff    |
    +-----+--------------------+---------------------+--------------------+
    | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
    | 2.0 | 5.2272727272727275 | -0.7954545454545454 | 5.0439752555137165 |
    | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
    +-----+--------------------+---------------------+--------------------+
    """
    semivariogram_stats = EmpiricalVariogram(
        input_array=input_array,
        step_size=step_size,
        max_range=max_range,
        weights=weights,
        direction=direction,
        tolerance=tolerance,
        is_semivariance=True,
        is_covariance=True,
        is_variance=True
    )
    return semivariogram_stats
```

## Class

Parts:

1. (Required) `Short description`,
2. (Optional) Class `summary`,
3. (Required) `Parameters` (if any is passed into class `__init__()`),
4. (Required) `Attributes` - only public.
5. (Required) `Methods` - only public and implemented dunder methods.
6. (Optional) `See Also` - links to the similar or important functions, classes, etc.
7. (Optional) `Notes` - equations and explanations.
8. (Required) `References` - if code is based on a specific publication or a work of someone else then it should be cited here.
9. (Required) `Examples` - example of usage.

### Template

```python

class GeoDataClass:
    """
    Class returns square of a given distance.
    
    Parameters
    ----------
    distance : float
               Real number. Distance from point A to point B.
               
    Attributes
    ----------
    dist : float
           Distance from point A to point B.
           
    y : int, constant=2
        Constant power factor.
        
    dist_square : float
                  Positive real number. Distance raised to the power of 2.
                   
    Methods
    -------
    square(pdist=None)
        If pdist is None: raises attribute dist to the power of 2.
        If pdist is given and it is a real number: overrides dist and dist_square attributes and raises pdist to the power of 2.
    
    __str__()
        Prints current state of the object.
        
    Example
    -------
    >>> geo = GeoDataClass(0)
    >>> geo.square(3)
    >>> print(geo)
    Square of 3 is 9.
    
    """
    
    def __init__(self, distance: float):
        self.dist = distance
        self.y = 2
        self.dist_square = self.square()
        
    def square(self, pdist=None):
        """
        Method raises dist to the power of 2 and stores result in dist_squared attribute.
        
        Parameters
        ----------
        pdist : float or None, default=None
                If given then it overrides dist and dist_squared attributes and a square of this number is returned.
        
        Returns
        -------
        float
            dist^2
        """
        if pdist is None:
            return self.dist ** self.y
        else:
            self.dist = pdist
            self.square()
        
    def __str__(self):
        return f'Square of {self.dist} is {self.dist_square}.'

```

### Example

```python
class EmpiricalVariogram:
    """
    Class calculates Experimental Semivariogram and Experimental Covariogram of a given dataset.

    Parameters
    ----------
    input_array : numpy array
                  coordinates and their values: (pt x, pt y, value) or (Point(), value).

    step_size : float
                distance between lags within each points are included in the calculations.

    max_range : float
                maximum range of analysis.

    weights : numpy array or None, optional, default=None
              weights assigned to points, index of weight must be the same as index of point, if provided then
              the semivariogram is weighted.

    direction : float (in range [0, 360]), optional, default=0
                direction of semivariogram, values from 0 to 360 degrees:
                * 0 or 180: is NS direction,
                * 90 or 270 is EW direction,
                * 45 or 225 is NE-SW direction,
                * 135 or 315 is NW-SE direction.

    tolerance : float (in range [0, 1]), optional, default=1
                If tolerance is 0 then points must be placed at a single line with the beginning in the origin of
                the coordinate system and the angle given by y axis and direction parameter. If tolerance is > 0 then
                the bin is selected as an elliptical area with major axis pointed in the same direction as the line
                for 0 tolerance.
                * The minor axis size is (tolerance * step_size)
                * The major axis size is ((1 - tolerance) * step_size)
                * The baseline point is at a center of the ellipse.
                Tolerance == 1 creates an omnidirectional semivariogram.

    is_semivariance : bool, optional, default=True
                      should semivariance be calculated?

    is_covariance : bool, optional, default=True
                    should covariance be calculated?

    is_variance : bool, optional, default=True
                  should variance be calculated?

    Attributes
    ----------
    input_array : numpy array
                  The array with coordinates and observed values.

    experimental_semivariance_array : numpy array or None, optional, default=None
                                      The array of semivariance per lag in the form:
                                      (lag, semivariance, number of points within lag).

    experimental_covariance_array : numpy array or None, optional, default=None
                                    The array of covariance per lag in the form:
                                    (lag, covariance, number of points within lag).

    experimental_semivariances : numpy array or None, optional, default=None
                                 The array of semivariances.

    experimental_covariances : numpy array or None, optional, default=None
                               The array of covariances, optional, default=None

    variance_covariances_diff : numpy array or None, optional, default=None
                                The array of differences c(0) - c(h).

    lags : numpy array or None, default=None
           The array of lags (upper bound for each lag).

    points_per_lag : numpy array or None, default=None
                     A number of points in each lag-bin.

    variance : float or None, optional, default=None
               The variance of a dataset, if data is second-order stationary then we are able to retrieve a semivariance
               as a difference between the variance and the experimental covariance:

                    (Eq. 1)

                        g(h) = c(0) - c(h)

                        where:

                        g(h): semivariance at a given lag h,
                        c(0): variance of a dataset,
                        c(h): covariance of a dataset.

                Important! Have in mind that it works only if process is second-order stationary (variance is the same
                for each distance bin) and if the semivariogram has the upper bound.
                See also: variance_covariances_diff attribute.

    step : float
        Derived from the step_size parameter.

    mx_rng : float
        Derived from the  max_range parameter.

    weights : numpy array or None
        Derived from the weights paramtere.

    direct: float
        Derived from the direction parameter.

    tol : float
        Derived from the tolerance parameter.

    Methods
    -------
    __str__()
        prints basic info about the class parameters.

    __repr__()
        reproduces class initialization with an input data.

    See Also
    --------
    calculate_covariance : function to calculate experimental covariance and variance of a given set of points.
    calculate_semivariance : function to calculate experimental semivariance from a given set of points.

    Examples
    --------
    >>> import numpy as np
    >>> REFERENCE_INPUT = np.array([
    ...    [0, 0, 8],
    ...    [1, 0, 6],
    ...    [2, 0, 4],
    ...    [3, 0, 3],
    ...    [4, 0, 6],
    ...    [5, 0, 5],
    ...    [6, 0, 7],
    ...    [7, 0, 2],
    ...    [8, 0, 8],
    ...    [9, 0, 9],
    ...    [10, 0, 5],
    ...    [11, 0, 6],
    ...    [12, 0, 3]
    ...    ])
    >>> STEP_SIZE = 1
    >>> MAX_RANGE = 4
    >>> empirical_smv = EmpiricalVariogram(REFERENCE_INPUT, step_size=STEP_SIZE, max_range=MAX_RANGE)
    >>> print(empirical_smv)
    +-----+--------------------+---------------------+--------------------+
    | lag |    semivariance    |      covariance     |    var_cov_diff    |
    +-----+--------------------+---------------------+--------------------+
    | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
    | 2.0 | 5.2272727272727275 | -0.7954545454545454 | 5.0439752555137165 |
    | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
    +-----+--------------------+---------------------+--------------------+
    """

    def __init__(self, input_array, step_size: float, max_range: float, weights=None, direction=0, tolerance=1,
                 is_semivariance=True, is_covariance=True, is_variance=True):

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        self.input_array = input_array
        self.experimental_semivariance_array = None
        self.experimental_covariance_array = None
        self.lags = None
        self.experimental_semivariances = None
        self.experimental_covariances = None
        self.variance_covariances_diff = None
        self.points_per_lag = None
        self.variance = 0

        self.step = step_size
        self.mx_rng = max_range
        self.weights = weights
        self.direct = direction
        self.tol = tolerance

        self.__c_sem = is_semivariance
        self.__c_cov = is_covariance
        self.__c_var = is_variance

    def _calculate_covariance(self, get_variance=False):
        """
        Method calculates covariance and variance.

        See : calculate_covariance function.
        """
        pass

    def _calculate_semivariance(self):
        """
        Method calculates semivariance.

        See: calculate_semivariance function.
        """
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def __str_empty(self):
        pass
```

## Modules

**Important!**

A description of modules in **Pyinterpolate** is slightly different than the `numpy` style. We include module docstring at a top of the file that groups multiple functions and classes and has own specific logic.

Parts:

1. (Required) `Short description`,
2. (Required) `Long description`,
3. (Required) `Changelog Table`: table with the major changes,
4. (Required) `Authors`: list with authors contributing to the module,
5. (Required) `References`: list of tutorials,
6. (Required) `Bibliography`: list of publications and articles related to the module.
7. (Optional) `TODO`: list of functions that should be created or debugged with links to the Github Issues.

The role of docstring in a module is to store information about changes, contributors and algorithm sources. We do not present examples - for this we have tutorials and function / class docs.

### Template

```python
"""
Module description is placed at the top of a file.

Here we describe module in a detail. The most important questions to answer here are:
- what problem does module solve?
- what methods does module incorporate to solve this problem?
- what kind of data flows into module?
- what is the output of module?
- why this module is important?
- relation to other modules.

Changelog
---------

| Date       | Change description        | Author         |
|------------|---------------------------|----------------|
| 2022-02-16 | First release of document | @SimonMolinsky |

Authors
-------
- Szymon Molinski @SimonMolinsky
- Peter Pan @PP-GithubProfileAlias-099990

References
----------
- [Docstrings in package](url)
- [Other tutorial](url)

Bibliography
------------
- Docstrings in numpy. Numpydoc maintaners. https://numpydoc.readthedocs.io/en/latest/format.html 2019

TODO
----
- Create a docstring for variogram.empirical module

"""
```

### Example

TODO :)

## Resources

1. [Numpydoc maintainers and contributors. Style guide. 2019](https://numpydoc.readthedocs.io/en/latest/format.html)
2. David Goodger, Guido van Rossum [Python PEP257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)