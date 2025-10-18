"""
Emergents: A population genetics and evolution simulation package.

This package provides tools for simulating genetic evolution, mutations,
and population dynamics in a high-performance, configurable framework.
"""

__author__ = "Jojobarbarr"
__version__ = "0.1.0"

# Import the logging configuration to ensure it's set up when package is imported
from emergents.logging_config import get_logger

# Package-level logger
logger = get_logger(__name__)
