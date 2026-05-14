"""
Utilities for configuring project loggers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path


VALID_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


@dataclass
class LoggerConfig:
    """
    Configuration for project loggers.

    Parameters
    ----------
    log_dir : str, default="logs"
        Directory used to save log files.
    log_level : str, default="INFO"
        Logging level, such as DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    log_to_console : bool, default=True
        Whether to print logs to the console.
    log_to_file : bool, default=True
        Whether to save logs to a file.
    """
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.log_dir, str) or not self.log_dir.strip():
            raise ValueError(
                f"Invalid log_dir={self.log_dir!r}. Expected a non-empty string."
            )

        if not isinstance(self.log_level, str):
            raise ValueError(
                f"Invalid log_level={self.log_level!r}. Expected a string."
            )

        normalized_level = self.log_level.upper()
        if normalized_level not in VALID_LOG_LEVELS:
            valid_levels = ", ".join(VALID_LOG_LEVELS)
            raise ValueError(
                f"Invalid log_level={self.log_level!r}. Expected one of: {valid_levels}."
            )
        self.log_level = normalized_level

        if not isinstance(self.log_to_console, bool):
            raise ValueError(
                f"Invalid log_to_console={self.log_to_console!r}. Expected a bool."
            )

        if not isinstance(self.log_to_file, bool):
            raise ValueError(
                f"Invalid log_to_file={self.log_to_file!r}. Expected a bool."
            )

        if not (self.log_to_console or self.log_to_file):
            raise ValueError(
                "Invalid logger output flags: "
                f"log_to_console={self.log_to_console!r}, "
                f"log_to_file={self.log_to_file!r}. At least one must be True."
            )

def _get_log_level(log_level: str) -> int:
    normalized_level = log_level.upper()
    level = VALID_LOG_LEVELS.get(normalized_level)

    if level is None:
        valid_levels = ", ".join(VALID_LOG_LEVELS)
        raise ValueError(
            f"Invalid log_level={log_level!r}. Expected one of: {valid_levels}."
        )

    return level

def get_logger(name: str, config: LoggerConfig | None = None) -> logging.Logger:
    """
    Create a configured project logger.

    Parameters
    ----------
    name : str
        Logger name, usually the module or script name.
    config : LoggerConfig, optional
        Logger configuration.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if config is None:
        config = LoggerConfig()

    logger = logging.getLogger(name)
    level = _get_log_level(config.log_level)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.log_to_file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger_file_name = name.replace(".", "")
        log_file = log_dir / f"{logger_file_name}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger

if __name__ == "__main__":
    config = LoggerConfig(log_dir="logs", log_level="INFO")

    logger = get_logger("logger_smoke_test", config)
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    same_logger = get_logger("logger_smoke_test", config)
    same_logger.info("This message should appear only once.")