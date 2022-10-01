Quickstart
==========

Installation
------------

Install package with `conda`:

.. code-block:: console

   conda install -c conda-forge pyinterpolate

Ordinary Kriging
----------------

The package has multiple spatial interpolation functions. The flow of analysis is usually the same for each method.
The interpolation of missing value from points is the basic case. We use for it *Ordinary Kriging*.

**[1.] Read and prepare data.**

.. code-block:: python

   from pyinterpolate import read_txt


   point_data = read_txt('dem.txt')


**[2.] Analyze data, calculate the experimental variogram.**

.. code-block:: python

   from pyinterpolate import build_experimental_variogram


   search_radius = 500
   max_range = 40000

   experimental_semivariogram = build_experimental_variogram(input_array=point_data,
                                                             step_size=search_radius,
                                                             max_range=max_range)

**[3.] Data transformation, fit theoretical variogram.**

.. code-block:: python

   from pyinterpolate import build_theoretical_variogram


   semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram,
                                         model_type='spherical',
                                         sill=400,
                                         rang=20000,
                                         nugget=0)

**[4.] Interpolation.**

.. code-block:: python

   from pyinterpolate import kriging


   unknown_point = (20000, 65000)
   prediction = kriging(observations=point_data,
                        theoretical_model=semivar,
                        points=[unknown_point],
                        how='ok',
                        no_neighbors=32)

**[5.] Error and uncertainty analysis.**

.. code-block:: python

   print(prediction)  # [predicted, variance error, lon, lat]

.. code-block:: console

   >> [211.23, 0.89, 20000, 60000]

**[6.] Full code.**

.. code-block:: python

    from pyinterpolate import read_txt
    from pyinterpolate import build_experimental_variogram
    from pyinterpolate import build_theoretical_variogram
    from pyinterpolate import kriging


    point_data = read_txt('dem.txt')  # x, y, value
    search_radius = 500
    max_range = 40000

    experimental_semivariogram = build_experimental_variogram(input_array=point_data,
                                                              step_size=search_radius,
                                                              max_range=max_range)
    semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram,
                                          model_type='spherical',
                                          sill=400,
                                          rang=20000,
                                          nugget=0)
    unknown_point = (20000, 65000)
    prediction = kriging(observations=point_data,
                         theoretical_model=semivar,
                         points=[unknown_point],
                         how='ok',
                         no_neighbors=32)

