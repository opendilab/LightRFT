"""
Module for extracting version information from pyproject.toml files.

This module provides utilities to read and parse pyproject.toml files to extract
version information. It handles compatibility across different Python versions
by using the appropriate TOML parsing library (tomllib for Python 3.11+ or
tomli for earlier versions).

The module is particularly useful for build scripts, CI/CD pipelines, or any
automation that needs to programmatically access the project version defined
in pyproject.toml.
"""

import sys

# Python 3.11+ has built-in tomllib
if sys.version_info >= (3, 11):
    import tomllib
else:
    # For Python 3.10, need to install tomli: pip install tomli
    try:
        import tomli as tomllib
    except (ModuleNotFoundError, ImportError):
        raise RuntimeError(
            "Missing required TOML parsing library. "
            "For Python versions before 3.11, you need to install the 'tomli' library. "
            "Please run: pip install tomli\n"
            "This library is required to parse pyproject.toml files since Python's built-in "
            "tomllib module is only available in Python 3.11 and later versions."
        )


def get_version_from_pyproject():
    """
    Extract the version string from the pyproject.toml file.

    This function reads the pyproject.toml file from the current working directory
    and extracts the version information from the [project] section. The function
    assumes the pyproject.toml file follows the PEP 621 standard format.

    :return: The version string as defined in pyproject.toml
    :rtype: str
    :raises FileNotFoundError: If pyproject.toml file is not found in the current directory
    :raises KeyError: If the version field is not found in the expected location
    :raises tomllib.TOMLDecodeError: If the pyproject.toml file contains invalid TOML syntax

    Example::

        >>> # Assuming pyproject.toml contains:
        >>> # [project]
        >>> # version = "1.0.0"
        >>> version = get_version_from_pyproject()
        >>> print(version)  # Output: "1.0.0"

    Note:
        The function expects the version to be located at data["project"]["version"]
        in the TOML structure, which follows the PEP 621 specification for
        Python project metadata.
    """
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


print(get_version_from_pyproject())
