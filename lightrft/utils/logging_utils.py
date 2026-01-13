"""Logging configuration using loguru."""
from typing import TYPE_CHECKING
from loguru import logger
import sys

if TYPE_CHECKING:
    from loguru import Logger

# Configure loguru with format similar to the old logging configuration
_FORMAT = (
    "<level>{level: <8}</level> <green>{time:MM-DD HH:mm:ss}</green> "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan>] {message}"
)

# Remove default handler and add custom one
logger.remove()


def init_logger(name: str, level: str = "DEBUG") -> "Logger":
    """
    Return the loguru logger instance.

    Note: loguru uses a singleton pattern, so all loggers share the same configuration.
    The 'name' parameter is kept for backward compatibility but is not used by loguru.

    :param name: Logger name (kept for backward compatibility)
    :type name: str
    :param level: Logging level (kept for backward compatibility)
    :type level: str
    :return: The loguru logger instance
    :rtype: loguru.Logger
    """
    logger.add(
        sys.stdout,
        format=_FORMAT,
        level=level,
        colorize=True,
    )
    return logger
