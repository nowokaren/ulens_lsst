"""
Sphinx configuration for ulens_lsst documentation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ulens_lsst', 'ulens_lsst')))
sys.path.insert(0, '/opt/lsst/software/stack/conda/envs/lsst-scipipe-10.0.0/lib/python3.12/site-packages')
# autodoc_mock_imports = ['lsst.afw', 'lsst.afw.detection', 'lsst.afw.detection.ImagePsf', 'lsst.afw.detection.Psf', 'lsst.daf.butler', 'lsst.daf', 'lsst.geom', 'lsst.utils']
autodoc_mock_imports = [
    'lsst',  # Covers lsst.afw, lsst.geom, etc.
    'rubin_sim',  # For rubin_sim imports
    'rubin_sim.maf',
    'photutils',  # Your custom bandpass module
]
autodoc_typehints = "none"  # Avoid C++ type issues
suppress_warnings = ['autodoc.import_object']  # Suppress import warnings

project = 'Microlensing LSST Light curve Simulator'
copyright = '2025, Karen Nowogrodzki'
author = 'Karen Nowogrodzki'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', 
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/*.ipynb.checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"

# Autodoc settings
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Intersphinx for external references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# nbsphinx settings
nbsphinx_execute = 'never'  # Avoid executing notebooks during build