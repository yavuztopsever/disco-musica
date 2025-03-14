"""
Logging utilities for Disco Musica.

This module provides functions for setting up and configuring logging.
"""

import os
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_level=logging.INFO, log_file=None, log_to_console=True):
    """
    Configure the logging system for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None)
        log_to_console: Whether to log to console (default: True)
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        os.makedirs(log_path.parent, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create a default log file if not specified
    elif log_to_console:
        # Get app data directory
        app_dir = Path.home() / ".disco-musica"
        os.makedirs(app_dir, exist_ok=True)
        
        log_path = app_dir / "disco-musica.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)