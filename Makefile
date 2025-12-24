.PHONY: docs test unittest resource

PYTHON := $(shell which python)

PROJ_DIR      := .
DOC_DIR       := ${PROJ_DIR}/docs
BUILD_DIR     := ${PROJ_DIR}/build
DIST_DIR      := ${PROJ_DIR}/dist
TEST_DIR      := ${PROJ_DIR}/test
TESTFILE_DIR  := ${TEST_DIR}/testfile
SRC_DIR       := ${PROJ_DIR}/lightrft

RANGE_DIR      ?= .
RANGE_TEST_DIR := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR  := ${SRC_DIR}/${RANGE_DIR}

COV_TYPES ?= xml term-missing

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
clean:
	rm -rf ${DIST_DIR} ${BUILD_DIR} *.egg-info

docs:
	$(MAKE) -C "${DOC_DIR}" html
docs-live:
	$(MAKE) -C "${DOC_DIR}" live

black:
	black --line-length=120 "${SRC_DIR}"
yapf:
	yapf --recursive --in-place "${SRC_DIR}"
flake8:
	flake8 --ignore=F403,F405,W504,W503,E203 --max-line-length=120 "${SRC_DIR}"
pylint:
	pylint --rcfile=.pylintrc --disable=C0114,C0415,W0212,W0235,W0238,W0621,C0103,R1735,C2801,E0402,C0412,W0719,R1728,W1514,W0718,W0105,W0707,C0209,W0703,W1203 "${SRC_DIR}"

#format: black
format: yapf
fcheck: flake8 pylint
