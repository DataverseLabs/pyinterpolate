# Installation

## Conda

Package is working on every operating system if you install it from the `conda` package manager with a command:

```shell
conda install -c conda-forge pyinterpolate
```

Conda installation requires Python in version >= 3.8.

## Pip

To install pyinterpolate from `pip` type in your terminal:

```
pip install pyinterpolate
```

## Q&A

### Jupter Notebook

*****

_I'd like to run Jupyter Notebooks, what should I do?_

*****

Install `pyinterpolate`  along `notebook` in your conda environment:

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
conda install -c conda-forge notebook pyinterpolate
```

Now you are able to run library from conda notebooks.


### The libspatialindex_c.so dependency error

*****

_libspatialindex_c.so dependency error_

*****

With Python==3.7 installation `rtree` and `GeoPandas` that are requirements for pyinterpolate may be not installed properly
because your operating system does not have `libspatialindex_c.so` file. In this case install it from terminal:

LINUX:

```
sudo apt install libspatialindex-dev
```

MAC OS:

```
brew install spatialindex
```
