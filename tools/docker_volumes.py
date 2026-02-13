"""
Docker volume argument construction utilities.

This module provides a small command-line utility for converting a semicolon-
separated list of Docker volume mappings into the ``-v`` argument format used
by the ``docker`` CLI. It is intended for lightweight scripting and environment-
variable-based configuration in automation workflows.

The module contains the following main components:

* :func:`main` - Command-line entry point that reads input and prints output

Example::

    >>> # Example CLI usage
    >>> # python -m tools.docker_volumes -i "/data:/data;/logs:/logs"
    >>> # Output: -v /data:/data -v /logs:/logs

    >>> # Example using the environment variable
    >>> # DOCKER_VOLUME_STR="/data:/data;/logs:/logs" python -m tools.docker_volumes
    >>> # Output: -v /data:/data -v /logs:/logs
"""

import argparse
import os
import sys
from typing import Iterable, Optional


def _build_volume_args(volume_str: str) -> str:
    """
    Build Docker ``-v`` arguments from a semicolon-delimited volume string.

    The input string is split on semicolons (``;``). Each non-empty segment is
    stripped and formatted as ``-v <segment>``. The resulting segments are
    joined with spaces to form a CLI-ready argument string.

    :param volume_str: Semicolon-delimited volume mapping string
    :type volume_str: str
    :return: Space-separated Docker ``-v`` arguments
    :rtype: str

    Example::

        >>> _build_volume_args("/data:/data;/logs:/logs")
        '-v /data:/data -v /logs:/logs'
    """
    segments = [seg.strip() for seg in volume_str.split(";") if seg.strip()]
    return " ".join(f"-v {seg}" for seg in segments)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    Parse command-line input and output Docker volume arguments.

    The function reads a semicolon-separated volume mapping string from the
    ``-i/--input`` command-line argument or from the ``DOCKER_VOLUME_STR``
    environment variable if the argument is not provided. If no input is
    available, the function returns ``0`` without writing output.

    :param argv: Optional iterable of command-line arguments, defaults to ``None``
    :type argv: Iterable[str] or None
    :return: Exit status code (always ``0`` for normal execution)
    :rtype: int

    Example::

        >>> main(["-i", "/data:/data;/logs:/logs"])
        0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="volume_str")
    args = parser.parse_args(argv)

    volume_str = args.volume_str or os.environ.get("DOCKER_VOLUME_STR", "")
    if not volume_str:
        return 0

    output = _build_volume_args(volume_str)
    if output:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
