# Data Structure

## VERSIONS >= 0.3

### DATA

Data structures are the input data types, sometimes with a specific constraints (column name, or position of value).

#### POINTS

Points are basic data structures.

#### BLOCKS

#### POINT-SUPPORT

### MODEL

#### EXPERIMENTAL VARIOGRAM

#### THEORETICAL VARIOGRAM

#### AGGREGATED VARIOGRAM

## VERSIONS < 0.3

Library works with geospatial datasets but they could be different from project to project. To ensure stable calculations specific data structures must be preserved by all scripts. Here we have gathered all data structures used by the library:

### POINTS:

Points are usually described only by their coordinates and value measured at a given location:

> [coordinate x, coordinate y, value] --> [float, float, float or int]


### AREAS:

Areas (polygons) are more complex than points and they are described by their id (optional parameter), geometry (shapely.geometry.polygon.Polygon), centroid, value:

> [area id name, [geometry ... ... ... ... ...], [centroid coordinate x, centroid coordinate y], value] --> [str or int, Polygon, list, float or int]


### POINTS WITHIN AREA:

Points within area are described by the area id and a list of all points coordinates and their values:

> [area id name, [[coordinate x, coordinate y, value], ..., [coordinate x, coordinate y, value]]]
