# pyinterpolate
Bunch of spatial interpolation scripts written in numpy and Python

Project pyinterpolate will aggregate GIS spatial interpolation scripts written in numpy and Python 3.6.

### Actual scope of work:

#### calculate_semivariogram(points_set, lags, step_size)

<b>Function calculates semivariance of points in n-dimensional space.</b>

* INPUT[0]: points_array: numpy array of points and values (especially DEM) where points_array[0] = array([point_x, point_y, ..., point_n, value])
* INPUT[1]: lags: array of lags between points
* INPUT[2]: step_size: distance which should be included in the gamma parameter which enhances range of interest
* OUTPUT: semivariance: numpy array of pair of lag and semivariance values where semivariance[0] = array([lag(i), semivariance for lag(i)])

-----

### Done:

#### calculate_distance(points_set)

<b>Python function for calculating a distance between points in n-dimensional space.</b>

* INPUT: points_set - numpy array with points' coordinates where each column indices new dimension 
* OUTPUT: distances - numpy array with euclidean distances between all pairs of points.
IMPORTANT! If input array size has <b>x</b> rows (coordinates) then output array size is x(cols) by x(rows) and each row describes distances between coordinate from row(i) with all rows. The first column in row is a distance between coordinate(i) and coordinate(0), the second row is a distance between coordinate(i) and coordinate(1) and so on.

-----

If you have any comments related to the repo feel free to contact me: s.molinski@datalions.eu
