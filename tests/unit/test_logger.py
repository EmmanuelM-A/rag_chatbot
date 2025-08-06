"""
Contains all the unit tests for logger function.
"""

import unittest
import logging
import os
from io import StringIO
import tempfile

from src.components.config.logger import get_logger
from src.components.config.settings import settings


class TestLogger(unittest.TestCase):
    """
    The test suite for the logger.
    """

    def setUp(self):
        """
        Common setup for all test cases.
        """
        self.logger_name = "test_logger"
        self.logger = get_logger(self.logger_name)
        self.stream = StringIO()

        self.stream_handler = logging.StreamHandler(self.stream)
        self.stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.stream_handler)

    def tearDown(self):
        """
        Cleanup after each test case.
        """
        self.logger.removeHandler(self.stream_handler)
        self.stream_handler.close()
        self.stream.close()

    def get_log_output(self):
        self.stream_handler.flush()
        return self.stream.getvalue()

    def test_debug_logged(self):
        self.logger.debug("Debug message test")
        self.assertIn("Debug message test", self.get_log_output())

    def test_info_logged(self):
        self.logger.info("Info message test")
        self.assertIn("Info message test", self.get_log_output())

    def test_warning_logged(self):
        self.logger.warning("Warning message test")
        self.assertIn("Warning message test", self.get_log_output())

    def test_error_logged(self):
        self.logger.error("Error message test")
        self.assertIn("Error message test", self.get_log_output())

    def test_critical_logged(self):
        self.logger.critical("Critical message test")
        self.assertIn("Critical message test", self.get_log_output())

    def test_log_to_console(self):
        self.logger.info("Console test")
        output = self.get_log_output()
        self.assertIn("Console test", output)

    def test_log_to_file_if_enabled(self):
        if not settings.IS_FILE_LOGGING_ENABLED:
            self.skipTest("File logging is disabled in settings.")

        with tempfile.NamedTemporaryFile(delete=False, mode='r+', encoding='utf-8') as tmpfile:
            log_file_path = tmpfile.name

        # Reconfigure logger to log to our temp file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        self.logger.info("File log test")

        file_handler.flush()
        self.logger.removeHandler(file_handler)
        file_handler.close()

        with open(log_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            self.assertIn("File log test", file_content)

        os.remove(log_file_path)

    def test_multiple_handlers(self):
        second_stream = StringIO()
        second_handler = logging.StreamHandler(second_stream)
        second_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(second_handler)

        self.logger.info("Message for both handlers")

        self.stream_handler.flush()
        second_handler.flush()

        self.assertIn("Message for both handlers", self.stream.getvalue())
        self.assertIn("Message for both handlers", second_stream.getvalue())

        self.logger.removeHandler(second_handler)
        second_handler.close()
        second_stream.close()

    def test_invalid_log_level(self):
        with self.assertRaises(ValueError):
            # Simulate a logger misconfiguration with invalid level
            invalid_logger = logging.getLogger("invalid_logger")
            invalid_logger.setLevel("INVALID")  # This won't raise, but handler level will
            handler = logging.StreamHandler()
            handler.setLevel("INVALID")  # Raises ValueError
            invalid_logger.addHandler(handler)

    def test_duplicate_logs_not_emitted(self):
        # Check logger does not emit duplicate messages if handlers are misconfigured
        self.logger.info("Duplicate check")
        first_output = self.get_log_output()

        self.logger.info("Duplicate check")
        second_output = self.get_log_output()

        # Should not double the same line
        self.assertEqual(first_output.count("Duplicate check"), 1)
        self.assertEqual(second_output.count("Duplicate check"), 1)

    def test_logger_configuration(self):
        """
        Tests logger-related settings.
        """
        self.assertIsInstance(settings.IS_FILE_LOGGING_ENABLED, bool)
        self.assertIsInstance(settings.LOG_DIRECTORY, str)
        self.assertIsInstance(settings.LOG_LEVEL, str)
