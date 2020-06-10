
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "itembed"
# copyright = "2020, Johan Berdat"
author = "Johan Berdat"

release = '0.4.2'

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
]

html_theme = "sphinx_rtd_theme"
html_show_copyright = False
