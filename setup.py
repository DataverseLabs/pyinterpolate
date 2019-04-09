import re
from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Project description
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

try:
    # obtain version string from __init__.py
    with open(path.join(here, 'pyinterpolate', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''

setup(
    name='pyinterpolate',
    version=version,
    description='Kriging interpolation Python module',
    url='https://github.com/szymon-datalions/pyinterpolate',
    author='Szymon Moliński | Data Lions',
    author_email='s.molinski@datalions.eu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: GIS specialists',
        'Topic :: Geoinformatics :: Spatial Interpolation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='Kriging Poisson Interpolation',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'tutorials']),
    install_requires=['matplotlib', 'numpy', 'geopandas'],
    project_urls={
        'Bug Reports': 'https://github.com/szymon-datalions/pyinterpolate/issues',
        'Sponsor page': 'http://datalions.eu/',
        'Source': 'https://github.com/szymon-datalions/pyinterpolate/',
    },
)
