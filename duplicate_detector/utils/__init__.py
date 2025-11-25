"""Utility modules for duplicate detector."""

from duplicate_detector.utils.logger import get_logger, StageLogger, PrintToLogger
from duplicate_detector.utils.performance import (
    profile_context,
    memory_tracker,
    PerformanceBenchmark,
    optimize_config_for_speed,
    optimize_config_for_accuracy
)

__all__ = [
    'get_logger',
    'StageLogger',
    'PrintToLogger',
    'profile_context',
    'memory_tracker',
    'PerformanceBenchmark',
    'optimize_config_for_speed',
    'optimize_config_for_accuracy'
]
