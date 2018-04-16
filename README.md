# pyinterpolate
Bunch of spatial interpolation scripts written in numpy and Python

Project pyinterpolate will aggregate GIS spatial interpolation scripts written in numpy and Python 3.6.

### Actual scope of work:

<b>Python function for calculating a distance between points in n-dimensional space</b>

/// <i><b>function calculate_distance(points_set)</i></b> ///

* INPUT: points_set - numpy array with points' coordinates where each column indices new dimension 
* OUTPUT: distances - numpy array with euclidian distances between all pairs of points.
IMPORTANT! If input array size has <b>x</b> rows (coordinates) then output array size is x(cols) by x(rows) and each row describes distances between coordinate from row(i) with all rows. The first column in row is a distance between coordinate(i) and coordinate(0), the second row is a distance between coordinate(i) and coordinate(1) and so on.

----------

If you have any comments related to the repo feel free to contact me: s.molinski@datalions.eu
