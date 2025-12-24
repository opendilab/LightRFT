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
# copyright = '2025, OpenDILab'
# author = 'OpenDILab'
copyright = ''
author = ''
release = 'v0.0.1'

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

html_theme_options = {
    # The target url that the logo directs to. Unset to do nothing
    'logo_url': 'https://mmocr.readthedocs.io/en/latest/',
    # "menu" is a list of dictionaries where you can specify the content and the
    # behavior of each item in the menu. Each item can either be a link or a
    # dropdown menu containing a list of links.
    'menu': [
        # A link
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/'
        },
        # A dropdown menu
        {
            'name': 'Projects',
            'children': [
                # A vanilla dropdown item
                {
                    'name': 'MMCV',
                    'url': 'https://github.com/open-mmlab/mmcv',
                },
                # A dropdown item with a description
                {
                    'name': 'MMDetection',
                    'url': 'https://github.com/open-mmlab/mmdetection',
                    'description': 'Object detection toolbox and benchmark'
                },
            ],
            # Optional, determining whether this dropdown menu will always be
            # highlighted.
            'active': True,
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

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
