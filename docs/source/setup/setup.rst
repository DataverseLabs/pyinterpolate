Setup
=====

Installation guidelines
-----------------------

The package can be installed from `conda` and `pip`. `Conda` installation requires Python >= 3.8, `pip` installation requires Python >= 3.7.

Conda
.....

.. code-block:: console

   conda install -c conda-forge pyinterpolate

pip
...

.. code-block:: console

   pip install pyinterpolate

Installation - additional topics
--------------------------------

Working with Notebooks
......................

Install `pyinterpolate`  along `notebook` in your conda environment:

**Step 1**:

.. code-block:: console

    conda create -n [NAME OF YOUR ENV]

**Step 2**:

.. code-block:: console

    conda activate [NAME OF YOUR ENV]

**Step 3**:

.. code-block:: console

    conda install -c conda-forge notebook pyinterpolate

Now you are able to run library from a notebook.

----

The libspatialindex_c.so dependency error
.........................................

With Python==3.7 installation `rtree` and `GeoPandas` that are requirements for pyinterpolate may be not installed properly
because your operating system does not have `libspatialindex_c.so` file. In this case install it from terminal:

**Linux**:

.. code-block:: console

    sudo apt install libspatialindex-dev

**MacOS**:

.. code-block:: console

    brew install spatialindex
