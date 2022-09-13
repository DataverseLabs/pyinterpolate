Setup
=====

Installation guidelines
-----------------------

|

Recommended - installation in conda environment
...............................................

**[1.]**

First, install system dependencies to use the package (``libspatialindex_c.so``):

**Linux**:

.. code-block:: console

   sudo apt install libspatialindex-dev

**MacOS**:

.. code-block:: console

   brew install spatialindex

----

**[2.]**

Next step is to create conda environment with Python >= 3.7. Recommended is Python 3.10. Additionally, we install ``pip`` and ``notebook`` packages, and then activate our environment:

.. code-block:: console

   conda create -n [YOUR ENV NAME] -c conda-forge python=3.10 pip notebook


.. code-block:: console

   conda activate [YOUR ENV NAME]

----

**[3.]**

In the next step, we install **pyinterpolate** and its dependencies with `pip` (conda distribution is under development and will be available soon):

.. code-block:: console

   pip install pyinterpolate

----

**[4.]**

You are ready to use the package!

----

pip installation
................

With **Python>=3.7** and system ```libspatialindex_c.so``` dependencies you may install package by simple command:

.. code-block:: console

   pip install pyinterpolate


A world of advice, you should use the *Virtual Environment* for the installation - every time and within every operating system. You may consider PipEnv too.

----

Installation - additional topics
--------------------------------

|

Working with Notebooks
......................

There is an additional step to run the package in Jupyter Notebooks. Before using ``pip`` you have to create ``conda``
environment and install required dependencies:

**Step 1**:

.. code-block:: console

    conda create -n [NAME OF YOUR ENV]

**Step 2**:

.. code-block:: console

    conda activate [NAME OF YOUR ENV]

**Step 3**:

.. code-block:: console

    conda install -c conda-forge pip notebook

**Step 4**:

.. code-block:: console

    pip install pyinterpolate

Now you are able to run library from a notebook.

----

The libspatialindex_c.so dependency error
.........................................

You could encounter the problem with **rtree** and **GeoPandas** packages installation. Both packages require a system dependency named **libspatialindex_c.so** and if it's missing, then you must install it manually.

**Linux**:

.. code-block:: console

    sudo apt install libspatialindex-dev

**MacOS**:

.. code-block:: console

    brew install spatialindex

----

How to install package within a virtual environment?
....................................................

*Coming soon...*
