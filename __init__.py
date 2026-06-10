# core/__init__.py



"""
app/core/logging.py
Centralised logging via loguru.
Import `logger` from here everywhere — never use print().
"""
import sys
from loguru import logger


def setup_logging(debug: bool = False) -> None:
    logger.remove()  # Remove default handler

    level = "DEBUG" if debug else "INFO"

    # Console — human-readable
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File — JSON structured, rotated daily, kept 7 days
    logger.add(
        "logs/app.log",
        level="INFO",
        rotation="1 day",
        retention="7 days",
        serialize=True,  # JSON lines
        enqueue=True,    # Thread-safe
    )

    logger.info("Logging initialised at level={}", level)


__all__ = ["logger", "setup_logging"]
