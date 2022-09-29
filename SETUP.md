Installation
------------

Package is working with linux and mac os systems. To install it download package and open it in the terminal then type:

```
pip install pyinterpolate
```

This command runs **setup.py** file inside package and install requirements from the list provided there.

*****

_I'd like to run Jupyter Notebooks, what should I do?_

*****

There is an additional step to run this library in Jupyter Notebooks. Before using pip you have to create conda
environment and install required dependencies:

#### Step 1:

```
conda create -n [NAME OF YOUR ENV]
```

#### Step 2:

```
conda activate [NAME OF YOUR ENV]
```

#### Step 3:

```
conda install -c conda-forge pip notebook
```

#### Step 4:

```
pip install pyinterpolate
```

Now you are able to run library from conda notebooks.

*****

_libspatialindex_c.so dependency error_

*****

Sometimes **rtree** (and / or **GeoPandas**) which are requirements for pyinterpolate may be not installed properly
because your operating system does not have **libspatialindex_c.so** file. In this case install it from terminal:

LINUX:

```
sudo apt install libspatialindex-dev
```

MAC OS:

```
brew install spatialindex
```

*****

_How to install package with virtual environment?_

*****

Coming soon...
