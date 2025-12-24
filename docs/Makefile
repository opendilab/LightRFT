# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXLIVE    = sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = build

# Parallel build option - use all available CPU cores for faster incremental builds
# Set to 'auto' to use all available cores, or specify a number (e.g., '4')
# Incremental builds are enabled by default (Sphinx caches doctrees)
PARALLEL      ?= auto

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
docs:
	@$(SPHINXBUILD) -j $(PARALLEL) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
live:
	@$(SPHINXLIVE) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
