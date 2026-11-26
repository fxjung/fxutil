from setuptools_scm import get_version

project = "fxutil"
copyright = "2025, Felix Jung"
author = "Felix Jung"
release = version = get_version(root="..", relative_to=__file__)

extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx_copybutton"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
autodoc_member_order = "bysource"
autodoc_typehints = "none"
