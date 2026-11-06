# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Build documentation with sphinx:
# sphinx-build -b html docs docs/_build/html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TOSTADA'
copyright = '2026, Martin Luther Universität'
author = 'Prerak Dhawan'
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
    # "autoapi.extension",
]

autodoc_mock_imports = ["meep", "cupy"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nb_execution_mode = "off"


myst_enable_extensions = ["dollarmath", "colon_fence"] # allow grids, etc. for md
myst_heading_anchors = 4
suppress_warnings = ["myst.header"]


# Auto API (deactivated, enable together with "autoapi.extension" above)
# autoapi_dirs = ["../src"]
# autoapi_root = "reference"
# autoapi_add_toctree_entry = False
# autoapi_keep_files = True
#
# autoapi_options = [
#     "members",
#     "undoc-members",
#     "show-inheritance",
#     "show-module-summary",
#     "special-members",
# ]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

add_module_names = False
toc_object_entries_show_parents = "hide"  # sidebar: "zoom" instead of "PointDistribution.zoom"
napoleon_use_ivar = True


def setup(app):
    # napoleon renders "Attributes" as the :ivar: field, which Sphinx labels "Variables"
    from sphinx.domains.python import PyObject

    for field in PyObject.doc_field_types:
        if field.name == "variable":
            field.label = "Attributes"

html_theme = 'sphinx_rtd_theme'
html_theme_options = {"navigation_depth": 6}  # default 4 hides the methods in the sidebar
html_static_path = ['_static']
html_css_files = ['custom.css']
