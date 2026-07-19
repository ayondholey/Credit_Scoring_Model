"""Logging configuration module."""
import sys
from pathlib import Path
from loguru import logger
from src.config import config


def setup_logging() -> None:
    """Configure logging for the application."""
    logger.remove()
    
    log_config = config.get_section("logging")
    
    logger.add(
        sys.stderr,
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
        colorize=True
    )
    
    log_file = Path(log_config.get("file", "logs/credit_scoring.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "10 days"),
        compression="zip"
    )
    
    logger.info("Logging configured successfully")


def get_logger(name: str):
    """Get logger instance with context."""
    return logger.bind(name=name)