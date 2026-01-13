import os
import sys

import pytorch_sphinx_theme

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Get current location
_DOC_PATH = os.path.dirname(os.path.abspath(__file__))
_PROJ_PATH = os.path.abspath(os.path.join(_DOC_PATH, '..', '..'))
_TP_PATH = os.path.abspath(os.path.join(_PROJ_PATH, 'third_party'))
_LIBS_PATH = os.path.join(_DOC_PATH, '_libs')
_SHIMS_PATH = os.path.join(_DOC_PATH, '_shims')
os.chdir(_PROJ_PATH)

# Set environment, remove the pre-installed package
sys.path.insert(0, _PROJ_PATH)
modnames = [mname for mname in sys.modules if mname.startswith('lightrft')]
for modname in modnames:
    del sys.modules[modname]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LightRFT'
copyright = '2025, OpenDILab'
author = 'OpenDILab'
release = 'v0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'enum_tools.autoenum',
    'nbsphinx',
    'sphinx_toolbox.collapse',
    'myst_parser',  # Support for Markdown files
]

# Support for both RST and Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",      # ::: style code fences
    "deflist",          # Definition lists
    "tasklist",         # Task lists with [ ] and [x]
    "substitution",     # Substitutions
    "linkify",          # Auto-detect URLs
]

templates_path = ['_templates']
exclude_patterns = []

# -- Incremental build configuration -----------------------------------------
# Sphinx supports incremental builds by default through doctree caching.
# The following settings optimize incremental compilation:

# Keep doctrees for incremental builds (default: True)
# This enables Sphinx to only rebuild changed files
keep_warnings = False  # Don't keep warnings in output for cleaner builds

# Nitpicky mode - set to False for faster incremental builds
# Set to True if you want strict checking of all references
nitpicky = False
nitpick_ignore = []

# Parallel build support (set via -j option in Makefile)
# Sphinx will automatically use parallel builds if available

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']
rtd_lang = 'en'

html_theme_options = {
    'logo': 'logo.png',
    'logo_url':
    'https://di-engine-docs.readthedocs.io/{}/latest/'.format(rtd_lang),
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/opendilab/LightRFT'
        },
    ],
    # For shared menu: If your project is a part of OpenMMLab's project and
    # you would like to append Docs and OpenMMLab section to the right
    # of the menu, you can specify menu_lang to choose the language of
    # shared contents. Available options are 'en' and 'cn'. Any other
    # strings will fall back to 'en'.
    'menu_lang':
        'en',
}

# -- Sidebar configuration -----------------------------------------
# Control which sidebar components are shown on each page
# Format: {document_pattern: [sidebar_templates]}
# 
# Common sidebar templates:
#   - 'globaltoc.html': Global table of contents (navigation tree)
#   - 'localtoc.html': Local table of contents (current page headings)
#   - 'relations.html': Previous/Next page navigation
#   - 'sourcelink.html': Link to source code
#   - 'searchbox.html': Search box
#
# Use '**' to match all documents, or specific patterns like 'index', 'api/*', etc.
#
# For pytorch_sphinx_theme, the default sidebar includes navigation.
# You can customize it per page type if needed.
html_sidebars = {
    # Default sidebar for all pages
    '**': [
        'globaltoc.html',  # Main navigation tree (controlled by toctree maxdepth)
        'localtoc.html',   # Current page's table of contents
        'relations.html',  # Previous/Next navigation
        'sourcelink.html', # Source code link
        'searchbox.html',  # Search box
    ],
    # Customize sidebar for specific pages if needed
    # 'index': ['globaltoc.html', 'searchbox.html'],  # Example: simpler sidebar for index
}

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
