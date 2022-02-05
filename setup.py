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
    download_url='https://github.com/DataverseLabs/pyinterpolate/archive/v0.2.tar.gz',
    author='Szymon Moliński',
    author_email='simon@ml-gis-service.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7 :: 3.8 :: 3.9 :: 3.10',
    ],
    keywords=['Spatial interpolation', 'Kriging', 'Area Kriging', 'Block Kriging', 'Poisson Kriging', 'Geostatistics'],
    packages=find_packages(exclude=['data', 'test', 'tutorials', 'new_concepts', 'paper', 'docs',
        'sample_data', 'developer']),

    install_requires=[],
    project_urls={
	'Webpage': 'https://pyinterpolate.com',
        'Bug Reports': 'https://github.com/szymon-datalions/pyinterpolate/issues',
        'Sponsor page': 'https://datalions.eu/',
        'Source': 'https://github.com/szymon-datalions/pyinterpolate/',
    },
)
