# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../../src'))

# print(sys.path)  # Check the Python path
# try:
#     # Try importing your module
#     from pyinterpolate.evaluate.cross_validation import validate_kriging
#     print("Import successful")
# except ImportError as e:
#     print(f"Import failed: {e}")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyinterpolate'
copyright = '2025, Szymon Moliński'
author = 'Szymon Moliński'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'numpydoc'
]

templates_path = ['_templates']
exclude_patterns = ['*__*__*', '**.ipynb_checkpoints']
add_module_names = True
# numpydoc_class_members_toctree = False
# autodoc_typehints = 'none'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
