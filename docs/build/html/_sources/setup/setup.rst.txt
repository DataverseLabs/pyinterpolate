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

Failing pylibtiff build - Linux
...............................

With installation on a fresh Linux distribution you might encounter problems with `pylibtiff` which is a dependency of `pyinterpolate`.
You will get a long error message. The last line of the message is:

.. code-block:: console

    ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (pylibtiff)

This means that you haven't configured required system libraries yet. You should install those three libraries:

- `libtiff-dev`
- `python3-dev`
- `gcc`

Install those using command:

.. code-block:: console

    sudo apt install gcc python3-dev libtiff-dev

and after installation you should be able to install `pyinterpolate` without problems.

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
