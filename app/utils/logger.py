"""Logging Configuration for the RAG Q&A System."""

from functools import lru_cache
import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the logging for the application.
    
    Args:
        log_level:logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    # create formatter
    formatter = logging.Formatter(
        fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging,log_level.upper(),logging.INFO))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to classes."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
