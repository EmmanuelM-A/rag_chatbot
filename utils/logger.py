"""
The centralized logging mechanism for the application.
"""

import logging
import os
from components.config import LOG_DIRECTORY


class ColorFormatter(logging.Formatter):
    """
    A log formatter that adds ANSI color codes to console output
    based on the log level.
    """

    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def get_logger(name: str, log_dir: str = LOG_DIRECTORY) -> logging.Logger:
    """
    Initializes and returns a logger with colorized console output and
    optional file logging.

    If the environment variable LOG_TO_FILE is set to "True",
    logs will also be written to 'app.log' and 'error.log'.

    Args:
        name (str): The logger name, usually __name__.
        log_dir (str): Directory where log files should be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger  # Avoid adding duplicate handlers

    logger.setLevel(logging.DEBUG)

    # Formatters
    formatter = logging.Formatter(
        '%(name)s -> %(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    color_formatter = ColorFormatter(
        '%(name)s -> %(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # Add file logging only if enabled
    if os.getenv("LOG_TO_FILE", "False").lower() == "true":
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        error_handler = logging.FileHandler(os.path.join(log_dir, "error.log"))
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    return logger
