"""Logging configuration using loguru."""
from loguru import logger
import sys

# Configure loguru with format similar to the old logging configuration
_FORMAT = (
    "<level>{level: <8}</level> <green>{time:MM-DD HH:mm:ss}</green> "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan>] {message}"
)

# Remove default handler and add custom one
logger.remove()
logger.add(
    sys.stdout,
    format=_FORMAT,
    level="DEBUG",
    colorize=True,
)


def init_logger(name: str):
    """
    Return the loguru logger instance.

    Note: loguru uses a singleton pattern, so all loggers share the same configuration.
    The 'name' parameter is kept for backward compatibility but is not used by loguru.

    :param name: Logger name (kept for backward compatibility)
    :type name: str
    :return: The loguru logger instance
    :rtype: loguru.Logger
    """
    return logger
