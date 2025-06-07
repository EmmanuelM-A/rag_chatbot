import logging
import os
from components.config import LOG_DIRECTORY


class ColorFormatter(logging.Formatter):
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


def get_logger(name: str, log_dir=LOG_DIRECTORY) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers

    logger.setLevel(logging.DEBUG)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Formatters
    formatter = logging.Formatter('%(name)s -> %(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    color_formatter = ColorFormatter('%(name)s -> %(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler (colorized)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # File handler for all logs
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # File handler for warnings and errors
    error_handler = logging.FileHandler(os.path.join(log_dir, "error.log"))
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger
