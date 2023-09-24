# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "frds"
copyright = "2023, Mingze Gao"
author = "Mingze Gao"
html_title = "frds"
html_shorttitle = "frds"
html_baseurl = "https://frds.io"
html_favicon = "images/frds.png"
html_logo = "images/frds_icon.png"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
html_css_files = [
    "css/custom.css",
]
exclude_patterns = []

toc_object_entries_show_parents = "hide"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Other configurations

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "autodoc_class_signature": "separate",
}

copybutton_exclude = ".linenos, .gp"

todo_include_todos = True
