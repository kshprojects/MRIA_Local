"""
MRIA - Medical Research Intelligence Assistant
Source package initialization
"""

__version__ = "1.0.0"
__author__ = "MRIA Development Team"
__description__ = "Medical Research Intelligence Assistant for Healthcare Professionals"

from .config import validate_environment
from .utils import setup_logging

# Initialize logging when package is imported
logger = setup_logging()

# Validate environment on import
try:
    validate_environment()
except SystemExit:
    logger.error("Environment validation failed. Please check your .env file.")
    raise