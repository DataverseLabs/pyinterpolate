import re
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

# Python 3.7 will be deprecated from the next version of a package
core_requirements = [
    'geopandas==0.11.1; python_version>="3.8"',
    'pandas==1.4.3; python_version>="3.8"',
    'numpy==1.23.1; python_version>="3.8"',
    'scipy==1.9.0; python_version>="3.8"',
    'tqdm==4.64.0; python_version>="3.8"',
    'descartes==1.1.0; python_version>="3.8"',
    'matplotlib==3.5.2; python_version>="3.8"',
    'prettytable==3.3.0; python_version>="3.8"',
    'rtree==1.0.0; python_version>="3.8"'
]

dask_requirements = [
    'dask==2022.2.1; python_version=="3.8" and sys_platform=="darwin"',
    'dask==2022.8.0; python_version>="3.9" and sys_platform=="darwin"',
    'dask==2022.8.0; python_version>="3.8" and sys_platform=="linux"'
]

python_37_linux_requirements = [
    'numpy==1.21.6; python_version=="3.7" and sys_platform=="linux"',
    'pandas==1.3.5; python_version=="3.7" and sys_platform=="linux"',
    'geopandas==0.10.2; python_version=="3.7" and sys_platform=="linux"',
    'descartes==1.1.0; python_version=="3.7" and sys_platform=="linux"',
    'tqdm==4.64.0; python_version=="3.7" and sys_platform=="linux"',
    'matplotlib==3.5.2; python_version=="3.7" and sys_platform=="linux"',
    'scipy==1.7.3; python_version=="3.7" and sys_platform=="linux"',
    'dask==2021.10.0; python_version=="3.7" and sys_platform=="linux"',
    'prettytable==3.3.0; python_version=="3.7" and sys_platform=="linux"',
    'rtree==1.0.0; python_version=="3.7" and sys_platform=="linux"'
]

python_37_macos_requirements = [
    'numpy==1.21.5; python_version=="3.7" and sys_platform=="darwin"',
    'pandas==1.3.5; python_version=="3.7" and sys_platform=="darwin"',
    'geopandas==0.10.2; python_version=="3.7" and sys_platform=="darwin"',
    'descartes==1.1.0; python_version=="3.7" and sys_platform=="darwin"',
    'tqdm==4.64.0; python_version=="3.7" and sys_platform=="darwin"',
    'matplotlib==3.5.1; python_version=="3.7" and sys_platform=="darwin"',
    'scipy==1.7.3; python_version=="3.7" and sys_platform=="darwin"',
    'dask==2021.10.0; python_version=="3.7" and sys_platform=="darwin"',
    'prettytable==3.3.0; python_version=="3.7" and sys_platform=="darwin"',
    'rtree==1.0.0; python_version=="3.7" and sys_platform=="darwin"'
]


requirements = core_requirements + dask_requirements + python_37_linux_requirements + python_37_macos_requirements

dev_requirements = {
    'dev': [
        'nbsphinx'
    ]
}

setup(
    name='pyinterpolate',
    version=version,
    description='Spatial interpolation Python module',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/szymon-datalions/pyinterpolate',
    download_url='https://github.com/DataverseLabs/pyinterpolate/archive/v0.3.tar.gz',
    author='Szymon Moliński',
    author_email='simon@dataverselabs.com',
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

    install_requires=requirements,
    extras_require=dev_requirements,
    project_urls={
        'Webpage': 'https://pyinterpolate.com',
        'Bug Reports': 'https://github.com/DataverseLabs/pyinterpolate/issues',
        'Source': 'https://github.com/DataverseLabs/pyinterpolate',
    },
)
