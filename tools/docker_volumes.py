"""
Docker volume argument construction utilities.

This module provides a lightweight command-line helper that converts a
semicolon-delimited Docker volume mapping string into a sequence of ``-v``
arguments suitable for direct use with the ``docker`` CLI.

The module contains the following public component:

* :func:`main` - Command-line entry point that reads input and prints output

.. note::
   The current implementation reads input from the ``DOCKER_VOLUME_STR``
   environment variable. Although an ``argv`` parameter exists for API
   compatibility and testing convenience, it is not currently parsed.

Example::

    >>> # Environment-variable-driven usage
    >>> # DOCKER_VOLUME_STR="/data:/data;/logs:/logs" python -m tools.docker_volumes
    >>> # Output: -v /data:/data -v /logs:/logs
"""

import os
import sys
from typing import Iterable, Optional


def _build_volume_args(volume_str: str) -> str:
    """
    Build Docker ``-v`` arguments from a semicolon-delimited volume string.

    The input string is split on semicolons (``;``). Each non-empty segment is
    stripped and converted into ``-v <segment>``. The resulting parts are joined
    with spaces into a CLI-ready argument string.

    :param volume_str: Semicolon-delimited Docker volume mapping string.
    :type volume_str: str
    :return: Space-separated Docker ``-v`` argument string.
    :rtype: str

    .. note::
       Empty segments and whitespace-only entries are ignored.

    Example::

        >>> _build_volume_args("/data:/data;/logs:/logs")
        '-v /data:/data -v /logs:/logs'
        >>> _build_volume_args(" /a:/a ; ; /b:/b ")
        '-v /a:/a -v /b:/b'
    """
    segments = [seg.strip() for seg in volume_str.split(";") if seg.strip()]
    return " ".join(f"-v {seg}" for seg in segments)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    Generate Docker volume CLI arguments and write them to standard output.

    This function reads a semicolon-separated volume mapping string from the
    ``DOCKER_VOLUME_STR`` environment variable. If the variable is empty or
    unset, the function exits successfully without producing output.

    The ``argv`` parameter is accepted for interface consistency and potential
    future extension, but it is not used by the current implementation.

    :param argv: Optional iterable of command-line arguments (currently unused).
    :type argv: Optional[Iterable[str]]
    :return: Process exit status code (``0`` for normal execution).
    :rtype: int

    .. note::
       Output is written directly to :data:`sys.stdout` without a trailing newline.

    Example::

        >>> import os
        >>> os.environ["DOCKER_VOLUME_STR"] = "/data:/data;/logs:/logs"
        >>> main([])
        0
    """
    _ = argv
    volume_str = os.environ.get("DOCKER_VOLUME_STR", "")
    if not volume_str:
        return 0

    output = _build_volume_args(volume_str)
    if output:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
