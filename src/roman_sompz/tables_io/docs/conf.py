# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from importlib.metadata import version

import autoapi

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tables_io"
author = "Eric Charles"
copyright = "2025, " + author
release = version("tables_io")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "myst_nb",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "tasklist"]
myst_heading_anchors = 3
copybutton_exclude = ".linenos, .gp, .go"


# autoapi set up
extensions.append("autoapi.extension")

templates_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

master_doc = "index"  # This assumes that sphinx-build is called from the root directory
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
add_module_names = False  # Remove namespaces from class/method signatures

autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_ignore = ["*/__main__.py", "*/_version.py"]
autoapi_add_toc_tree_entry = False
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "inherited-members",
]

html_theme = "sphinx_rtd_theme"
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]
html_theme_options = {"style_external_links": True}
