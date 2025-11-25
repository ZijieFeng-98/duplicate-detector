"""
Structured Logging System for Duplicate Detector

Provides centralized logging with:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File rotation
- Formatted output with timestamps
- Context managers for stage timing
- Integration with existing print statements
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import os
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Add color if outputting to terminal
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            log_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class StageLogger:
    """Context manager for logging pipeline stages with timing"""
    
    def __init__(self, logger: logging.Logger, stage_name: str, level: int = logging.INFO):
        self.logger = logger
        self.stage_name = stage_name
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting: {self.stage_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.stage_name} ({elapsed:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.stage_name} ({elapsed:.2f}s) - {exc_val}")
        return False  # Don't suppress exceptions


def setup_logger(
    name: str = "duplicate_detector",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Default log level
        console_level: Console log level (defaults to level)
        file_level: File log level (defaults to level)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level if console_level is not None else level)
    console_formatter = ColoredFormatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level if file_level is not None else level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name (defaults to 'duplicate_detector')
    
    Returns:
        Logger instance
    """
    if name is None:
        name = "duplicate_detector"
    
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class PrintToLogger:
    """
    Redirect print statements to logger
    
    Usage:
        print_to_logger = PrintToLogger(logger)
        print = print_to_logger  # Replace print function
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
    
    def __call__(self, *args, **kwargs):
        # Convert all arguments to strings
        message = ' '.join(str(arg) for arg in args)
        
        # Determine log level from emoji prefixes
        if message.startswith('✅') or message.startswith('✓'):
            self.logger.info(message)
        elif message.startswith('⚠️') or message.startswith('⚠'):
            self.logger.warning(message)
        elif message.startswith('❌') or message.startswith('❌'):
            self.logger.error(message)
        elif message.startswith('ℹ️') or message.startswith('ℹ'):
            self.logger.info(message)
        elif message.startswith('🔬') or message.startswith('📊'):
            self.logger.info(message)
        else:
            self.logger.log(self.level, message)


# Global logger instance (will be initialized on first use)
_global_logger: Optional[logging.Logger] = None


def initialize_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True
) -> logging.Logger:
    """
    Initialize global logging system
    
    Args:
        log_dir: Directory for log files (defaults to current directory)
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
    
    Returns:
        Logger instance
    """
    global _global_logger
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    log_file = None
    if log_to_file and log_dir:
        log_file = log_dir / "duplicate_detector.log"
    
    _global_logger = setup_logger(
        name="duplicate_detector",
        log_file=log_file,
        level=level
    )
    
    return _global_logger


def get_global_logger() -> logging.Logger:
    """Get the global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = initialize_logging()
    
    return _global_logger

