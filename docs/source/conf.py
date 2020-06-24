
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "itembed"
# copyright = "2020, Johan Berdat"
author = "Johan Berdat"

release = "0.5.0"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

html_theme = "sphinx_rtd_theme"
html_show_copyright = False
