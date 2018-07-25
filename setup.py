from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Project description
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyinterpolate',
    version='0.1.0',
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
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy==1.14.0', 'matplotlib==2.1.1'],
    project_urls={
        'Bug Reports': 'https://github.com/szymon-datalions/pyinterpolate/issues',
        'Sponsor page': 'http://datalions.eu/',
        'Source': 'https://github.com/szymon-datalions/pyinterpolate/',
    },
)