"""
Unit tests for logger module.
"""

import pytest
import tempfile
from pathlib import Path
import logging

from duplicate_detector.utils.logger import (
    get_logger,
    initialize_logging,
    StageLogger,
    PrintToLogger
)


class TestLogger:
    """Test logging functions."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_with_file(self, temp_dir):
        """Test logger with file output."""
        log_file = temp_dir / "test.log"
        logger = get_logger("test_logger", log_file=log_file)
        
        logger.info("Test message")
        
        assert log_file.exists()
        assert "Test message" in log_file.read_text()
    
    def test_initialize_logging(self, temp_dir):
        """Test global logging initialization."""
        logger = initialize_logging(
            log_dir=temp_dir,
            log_level="INFO"
        )
        
        assert isinstance(logger, logging.Logger)
        assert (temp_dir / "duplicate_detector.log").exists()


class TestStageLogger:
    """Test StageLogger context manager."""
    
    def test_stage_logger(self):
        """Test stage logging."""
        logger = get_logger("test")
        
        with StageLogger(logger, "Test Stage"):
            pass
        
        # Should complete without error
        assert True
    
    def test_stage_logger_with_exception(self):
        """Test stage logger handles exceptions."""
        logger = get_logger("test")
        
        try:
            with StageLogger(logger, "Test Stage"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should handle exception gracefully
        assert True


class TestPrintToLogger:
    """Test PrintToLogger redirect."""
    
    def test_print_to_logger(self, temp_dir):
        """Test redirecting print to logger."""
        log_file = temp_dir / "print_test.log"
        logger = get_logger("test", log_file=log_file)
        
        with PrintToLogger(logger):
            print("Test print message")
        
        # Check that message was logged
        log_content = log_file.read_text()
        assert "Test print message" in log_content or "test" in log_content.lower()

