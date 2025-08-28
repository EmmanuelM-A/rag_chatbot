"""
The centralized logging mechanism for the application. Provides clean console
output for interactive use.
"""

import logging
import os
import sys

from src.components.config.settings import settings


class ColorFormatter(logging.Formatter):
    """
    A log formatter that adds ANSI color codes to console output
    based on the log level.
    """

    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def __get_log_level(level: str) -> int:
    """
    Converts the log level string into its numerical counterpart.
    """

    level_mappings = {
        "CRITICAL": 50,
        "FATAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "WARN": 30,
        "INFO": 20,
        "DEBUG": 10
    }

    return level_mappings.get(level.upper(), 20)


def __get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Initializes and returns a default_logger that logs either to console OR files (not both).

    Args:
        name (str): The default_logger name, usually __name__.
        log_dir (str): Directory where log files should be stored (uses settings default if None).

    Returns:
        logging.Logger: Configured default_logger instance.
    """

    if log_dir is None:
        log_dir = settings.logging.LOG_DIRECTORY

    default_logger = logging.getLogger(name)

    if default_logger.hasHandlers():
        return default_logger  # Avoid adding duplicate handlers

    # Set log level from settings
    log_level = __get_log_level(settings.logging.LOG_LEVEL)
    default_logger.setLevel(log_level)

    # Create formatters
    file_formatter = logging.Formatter(
        fmt=settings.logging.LOG_FORMAT,
        datefmt=settings.logging.LOG_DATE_FORMAT
    )
    console_formatter = ColorFormatter(
        fmt=settings.logging.LOG_FORMAT,
        datefmt=settings.logging.LOG_DATE_FORMAT
    )

    if settings.logging.IS_FILE_LOGGING_ENABLED:
        # FILE LOGGING MODE
        try:
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # General log file (all logs)
            app_log_path = os.path.join(log_dir, "app.log")
            file_handler = logging.FileHandler(app_log_path, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(file_formatter)
            default_logger.addHandler(file_handler)

            # Error log file (errors only)
            error_log_path = os.path.join(log_dir, "error.log")
            error_handler = logging.FileHandler(error_log_path, encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            default_logger.addHandler(error_handler)
        except ValueError:
            # Fallback to console if file logging fails
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            default_logger.addHandler(console_handler)
    else:
        # CONSOLE LOGGING MODE
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        default_logger.addHandler(console_handler)

    return default_logger

logger = __get_logger(__name__)
