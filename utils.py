#!/usr/bin/env python3
"""
Utility module for shared functionality across the scraper project.
Contains logging configuration, formatters, and common constants.
"""

import logging
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Token calculation constant
CHARS_PER_TOKEN = 4.5


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.CYAN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        # Format the message
        result = super().format(record)
        # Reset levelname for next use
        record.levelname = levelname
        return result


def setup_logging(level=logging.INFO, format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    Configure logging with colored formatter.
    
    Args:
        level: Logging level (default: logging.INFO)
        format_string: Log message format string
    
    Returns:
        Configured logger instance
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(format_string))
    logging.basicConfig(
        level=level,
        handlers=[handler]
    )
    return logging.getLogger(__name__)
