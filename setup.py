import re
import sys
from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Project description
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# obtain version string from __init__.py
with open(path.join(here, '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search("__version__ = '(.*)'", init_py).groups()[0]

setup(
    name='pyinterpolate',
    version=version,
    description='Kriging interpolation Python module',
    url='https://github.com/szymon-datalions/pyinterpolate',
    author='Szymon Moliński | Data Lions',
    author_email='s.molinski@datalions.eu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: GIS specialists, Data Scientists, Geostatisticans',
        'Topic :: Geoinformatics :: Spatial Interpolation',
        'License :: OSI Approved :: BSD 3-Clause "New" or "Revised" License',
        'Programming Language :: Python :: 3.7.6',
    ],
    keywords='Kriging Spatial Analysis Ordinary Kriging Simple Kriging Poisson Kriging',
    packages=find_packages(exclude=['data', 'test', 'tutorials', 'new_concepts']),

    install_requires=['descartes==1.1.0', 'geopandas==0.7.0', 'matplotlib==3.2.1', 'numpy==1.18.3', 'tqdm==4.47.0',
                      'pyproj==2.6.0', 'scipy==1.4.1', 'shapely==1.7.0',
		      'fiona==1.8.13.post1; sys_platform=="darwin"', 
                      'rtree==0.9.4; sys_platform=="darwin"',
		      'fiona==1.8; sys_platform=="linux"',
                      'rtree>=0.8,<0.9; sys_platform=="linux"'],
    project_urls={
	'Webpage': 'https://pyinterpolate.com',
        'Bug Reports': 'https://github.com/szymon-datalions/pyinterpolate/issues',
        'Sponsor page': 'https://datalions.eu/',
        'Source': 'https://github.com/szymon-datalions/pyinterpolate/',
    },
)
