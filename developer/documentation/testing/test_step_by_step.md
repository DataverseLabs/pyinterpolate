# Tests - step by step tutorial

In this tutorial we will create *unit* and *functional* tests for the code developed within **pyinterpolate** package.

## Changelog

| Date | Change description | Author |
|--------|--------------------------|----------|
| 2021-06-23 | First release of tutorial | @szymon-datalions |

## Contents

- [Introduction](#introduction)
- [(Optional) Example function](#example-function)
- [Unit tests](#unit-tests)
- [Logical and functionality testing](#logical-and-functionality-testing)
- [Testing with tutorials](#test-with-tutorials)
- [(Optional) Tutorial as a test case](#tutorial-as-a-test)

## Introduction

Every time when code is changed or new feature is added we should perform tests to ensure that everything works as it should be. We have few different levels of test to perform. List below represents all steps required to be sure that everyting works (hopefully) fine. Steps 1-2 and 4 are required for the new features, steps 3 and 4 are required for changes in a codebase.

1. Write unit tests in `test` directory to check if input data structure is valid or if returned values are within specific or or are of specific type. Use `unittests` package. Group tests in regards to the module within you're working. As example if you implement new Kriging technique put your tests inside `test/kriging` module. Name your test file with prefix `test_` and follow `unittest` naming of the testing functions and classes. This part is of tutorial is presented in the section [Unit tests](#unit-tests).
2. Write logical test where you know exactly what should be returned. This step is very important in the development of scientific software so pen & paper are equally important as keybord & monitor. Where it's possible use examples from the literature and try to achieve the same results with the same datasets. If this is not possible justify why function is usable and why your results are different than in the literature. If you are not sure what it all means then read part about [Logical and functionality testing](#logical-and-functionality-testing).
3. Run all tests within the `test` package. You have two options: use `PyCharm` or Python console. Those are described in depth in the section [How to run multiple unit tests](#how-to-run-multiple-unit-tests).
4. [Create testing `conda` environment with implemented and tested functionality](#test-with-tutorials). Update all tutorials where change / addition may affect calculation results. Remember to update **Changelog table** in recalculated tutorials.
5. (Optional) [Write a tutorial](#tutorial-as-a-test) which covers your functionality or code change.

To make things easier to understand we will gou through the example of `calculate_seimvariance()` function.

## Example function

To start with the development we must first do two things:

1. Write an equation and / or create block diagram algorithm of the function,
2. Prepare dataset for logic tests.

Those two steps prepare our mental model. In the ideal situation we should have equation / algorithm blocks and sample data from publications. Fortunately for us experimental semivariogram calculation process is well described in the book **Basic Linear Geostatistics** written by *Margaret Armstrong* in 1998 (pages 47-52 for this tutorial). (If you're a geostatistican and you haven't read this book yet then do it as soon as you can. This resource is a gem among books for geostatistics and **Pyinterpolate** relies heavely on it).

Starting from the equation for experimental semivariogram:

$$\gamma'(h) = \frac{1}{2 * N(h)} \sum_{i=1}^{N(h)} [Z(x_{i} + h) - Z(x_{i})]^2$$

where $\gamma'(h)$ is a semivariance for a given lag $h$, $x_{i}$ are the locations of samples, $Z(x_{i})$ are their values and $N(h)$ is the number of pairs $(x_{i}, x_{i + h})$. What it means in practice? We may freely translate it to: **semivariance at a given interval of distances is a halved mean squared error of all points pairs which are in a given interval**. If we understand what it means then we could go further. To the block diagram of an algorithm. Here's a little digression: equations are not always available and sometimes we will implement processes and not single blocks. Take as an example genetic algorithms: they are very complex to desribe them with formal mathematical notation and we should consider text / diagram description in this case.

With a block diagram we may think of the first bunch of tests which should be implemented. Usually those tests check if an input data is valid and if results are valid. We may use algorithm presented in the page 49 in the **Basic Linear Geostatistics** but for the case of simplicity we create our own algorithm for omnidirectional semivariogram calculation. Algorithm works as follow:

(1) Read Data as an array of triplets `x, y, value`,

(2) Calculate distances between each element from the array,

(3) Create lags list to group semivariances (lags are separation distances $h$),

(4a) For each lag group calculated distances to select points within a specific range (lag),

(4b) Calculated mean squared error between all points pairs and divide it by two (each pair is taken twice a time),

(4c) Store calculated semivariance along the lag within array `lag, semivariance`,

(5) Return array of lags and their semivariances.

At this point we should see specific dependencies and data structures which should be tested. We start from the *unit tests*.

## Unit tests

We will create simple unit test which checks if all results are positive numbers. (Lags can be only positive because there is no negative distance. The same for semivariance - due to the fact that it is a squared difference it must always be a positive number). First, let's create Python file within `test/semivariance`. All files with unit tests should have a prefix `test_`. That's why we name our file `test_calculate_semivariance` and full path to the file should be:

`pyinterpolate/test/semivariance/test_calculate_semivariance.py`

At the beginning we must import `calculate_semivariance()` method and `unittest` module.

```python
import unittest
from pyinterpolate.semivariance import calculate_semivariance
```

To write a test we must create **class** which starts with `Test` prefix. Usually it is named `TestYourFunctionOrModuleName`, in our case: `TestCalculateSemivariance`. This class inherits from the `unittest.TestCase` class. We can skip explanation what inheritance is. The key is to understand that we can use methods from `unittest.TestCase` in our class `TestCalculateSemivariance`. Let's update our script with this new piece of information:

```python
import unittest
from pyinterpolate.semivariance import calculate_semivariance

class TestCalculateSemivariance(unittest.TestCase):

	def test_calculate_semivariance(self):
		pass
```

Good practice with unit testing is to have data which is not depended on the external sources or processes. In other words we use mostly static files with known datasets or artificial arrays. Those arrays may be filled with random numbers of specific distribution or hard-coded values which are simulating possible input. We should avoid using additional scripts from within the package if there is a chance to do so.



## Logical and functionality testing

## How to run multiple unit tests

## Test with tutorials

## Tutorial as a test