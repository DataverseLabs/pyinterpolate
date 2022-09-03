# Input Data Schema

`Pyinterpolate` works with two different kinds of interpolation scenarios, and both require specific data schema.

1. Point-based interpolation.
2. Area-based interpolation.

## Point interpolation

A point interpolation method uses an array of values and their locations. We define the core data schema as an array with three columns: `[x, y, value]` where the first and second are coordinates, and the third column represents a measurement. Our typical array may look like this:

```
[15.1152409 , 52.76514556, 91.27559662],
[15.1152409 , 52.74279035, 96.54829407],
[15.1152409 , 52.71070647, 51.25455093]
```

The first column is longitude, the second is latitude, and the third is a digital elevation model’s estimation. `Pyinterpolate` transforms each point input into **GeoDataFrame** with column `geometry` with `Point()` type geometry and column `value` with an observed value:

| **geometry** |**value** |
|----------|------|
| `POINT(15.11, 52.76)` | `91.27559662` |

We can skip reading data by methods included in `pyinterpolate.io` module and directly provide `GeoDataFrame` with geometry and values (only two columns with `Point()` and numeric data types). The important thing to notice is that we should provide a projection of our point array. If we use `read_txt()` or `read_csv()` functions and doesn’t provide `epsg` or `crs` then program uses EPSG:4326 as a default which can lead to the wrong results.

## Area interpolation

We could treat input data used for areal interpolation as a superset of point data. Why? Because the semivariogram regularization requires us to provide the point-support data along with standard polygons from `shapefile` or `geojson`. The complex object that consists of areas, points, areal aggregates and point measurements cannot be a simple array. Therefore we use `GeoDataFrame`. The core block data structure is a `GeoDataFrame` with columns:

- **id**: with area/block id,
- **geometry**: with polygon geometry,
- **value**: observations aggregated over an area at a specific period,
- **centroid**: area centroid, it is derived from the geometry if not given.

The example `GeoDataFrame` may look like this:

| **id** | **geometry** | **value** | **centroid** |
|----|----------|-------|----------|
| `aa` | `POLYGON(...)` | `100.0` | `POINT(17.71, 52,29)` |

To read area from a spatial file, we must now value column name and geometry column name. The name of the ID column is optional, the same for the centroid column, which is calculated from geometry anyway. We use `read_block()` function for it.

