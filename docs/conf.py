# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TOSTADA'
copyright = '2026, Dr. Prerak Dhawan'
author = 'Dr. Prerak Dhawan'
release = '0.4.0'

html_title = "Tostada Documentation"

root_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb", #jupyter notebooks
    "sphinx.ext.autodoc", # auto documentation
    "sphinx.ext.napoleon", # specific for our function documentation style
    "sphinx_design", # sphinx design for md files
    "autoapi.extension"
]

autodoc_mock_imports = ["meep", "cupy"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nb_execution_mode = "off"

myst_enable_extensions = ["dollarmath", "colon_fence"] # allow grids, etc. for md


# Auto API
autoapi_dirs = ["../src"]
autoapi_root = "reference"
autoapi_add_toctree_entry = False
autoapi_keep_files = True

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
