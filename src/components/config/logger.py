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
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[92m',    # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m' # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


class LoggerManager:
    """
    Singleton Logger Manager that ensures centralized configuration
    while allowing loggers to be accessed by name.
    """

    _instances: dict[str, logging.Logger] = {}

    @staticmethod
    def __get_log_level(level: str) -> int:
        """Converts the log level string into its numerical counterpart."""
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

    @classmethod
    def get_logger(
            cls,
            name: str = None,
            log_dir: str = settings.logging.LOG_DIRECTORY
    ) -> logging.Logger:
        """
        Returns a configured logger. Reuses existing loggers by name.

        Args:
            name (str): Logger name (e.g. __name__). Defaults to "app".
            log_dir (str): Directory where logs should be written (if file logging enabled).

        Returns:
            logging.Logger: Configured logger instance.
        """

        if not name:
            name = "app"

        if name in cls._instances:
            return cls._instances[name]

        default_logger = logging.getLogger(name)

        # Set log level
        log_level = cls.__get_log_level(settings.logging.LOG_LEVEL)
        default_logger.setLevel(log_level)

        # Avoid duplicate handlers
        if not default_logger.hasHandlers():
            file_formatter = logging.Formatter(
                fmt=settings.logging.LOG_FORMAT,
                datefmt=settings.logging.LOG_DATE_FORMAT
            )
            console_formatter = ColorFormatter(
                fmt=settings.logging.LOG_FORMAT,
                datefmt=settings.logging.LOG_DATE_FORMAT
            )

            if settings.logging.IS_FILE_LOGGING_ENABLED:
                try:
                    os.makedirs(log_dir, exist_ok=True)

                    # General log file (all logs)
                    app_log_path = os.path.join(log_dir, "app.log")
                    file_handler = logging.FileHandler(app_log_path, encoding="utf-8")
                    file_handler.setLevel(log_level)
                    file_handler.setFormatter(file_formatter)
                    default_logger.addHandler(file_handler)

                    # Error log file
                    error_log_path = os.path.join(log_dir, "error.log")
                    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
                    error_handler.setLevel(logging.ERROR)
                    error_handler.setFormatter(file_formatter)
                    default_logger.addHandler(error_handler)
                except Exception:
                    # Fallback to console
                    console_handler = logging.StreamHandler(sys.stdout)
                    console_handler.setFormatter(console_formatter)
                    default_logger.addHandler(console_handler)
            else:
                # Console logging
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(log_level)
                console_handler.setFormatter(console_formatter)
                default_logger.addHandler(console_handler)

        cls._instances[name] = default_logger
        return default_logger


# --- Global logger, dynamically switchable ---
logger: logging.Logger = LoggerManager.get_logger(__name__)

def set_logger(name: str) -> logging.Logger:
    """
    Dynamically change the global logger to use a different name.
    Returns the updated logger.
    """
    global logger
    logger = LoggerManager.get_logger(name)
    return logger
