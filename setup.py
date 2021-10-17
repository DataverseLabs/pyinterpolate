import re
import sys
from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Project description
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

# obtain version string from __init__.py
with open(path.join(here, '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search("__version__ = '(.*)'", init_py).groups()[0]

setup(
    name='pyinterpolate',
    version=version,
    description='Spatial interpolation Python module',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/szymon-datalions/pyinterpolate',
    download_url='https://github.com/szymon-datalions/pyinterpolate/archive/v0.2.tar.gz',
    author='Szymon Moliński',
    author_email='simon@ml-gis-service.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['Spatial interpolation', 'Kriging', 'Area Kriging', 'Block Kriging', 'Poisson Kriging', 'Geostatistics'],
    packages=find_packages(exclude=['data', 'test', 'tutorials', 'new_concepts', 'paper', 'docs',
        'sample_data', 'developer']),

    install_requires=[
        'descartes==1.1.0',
        'geopandas==0.9.0',
        'matplotlib==3.4.2',
        'numpy==1.20.3',
        'pyproj==3.1.0',
        'scipy==1.6.3',
        'shapely==1.7.1',
        'tqdm==4.61.0',
        'fiona==1.8.20; sys_platform=="darwin"',
        'rtree>0.9; sys_platform=="darwin"',
        'fiona==1.8; sys_platform=="linux"',
        'rtree>=0.8,<0.9; sys_platform=="linux"'],
    project_urls={
	'Webpage': 'https://pyinterpolate.com',
        'Bug Reports': 'https://github.com/szymon-datalions/pyinterpolate/issues',
        'Sponsor page': 'https://datalions.eu/',
        'Source': 'https://github.com/szymon-datalions/pyinterpolate/',
    },
)
