"""Module to create colour-formatted loggers for console output."""

import logging


class LoggerFactory:
    """Factory class for creating colour-formatted loggers."""

    COLORS = {
        "INFO": "\033[94m",  # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "DEBUG": "\033[90m",  # Grey
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    class ColorFormatter(logging.Formatter):
        """Custom formatter that adds colours based on log level."""

        def format(self, record: logging.LogRecord) -> str:
            """Formats a log record with colours based on its log level."""

            color = LoggerFactory.COLORS.get(record.levelname, LoggerFactory.RESET)
            message = super().format(record)
            return f"{color}{message}{LoggerFactory.RESET}"

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Returns a logger with colour formatting for console output."""

        formatter = LoggerFactory.ColorFormatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%H:%M"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False

        return logger
