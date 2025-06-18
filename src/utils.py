import logging
import sys
from typing import Any, Dict
from rich.console import Console
from rich.logging import RichHandler

# Initialize Rich console
console = Console()

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging with Rich formatting
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("MRIA")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create Rich handler
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=True
    )
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

def log_function_call(func_name: str, args: Dict[str, Any] = None, kwargs: Dict[str, Any] = None):
    """
    Log function calls for debugging
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
    """
    logger = logging.getLogger("MRIA")
    args_str = f"args={args}" if args else ""
    kwargs_str = f"kwargs={kwargs}" if kwargs else ""
    logger.debug(f"Calling {func_name}({args_str}, {kwargs_str})")

def handle_error(error: Exception, context: str = "") -> str:
    """
    Handle errors with consistent logging and formatting
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    
    Returns:
        Formatted error message
    """
    logger = logging.getLogger("MRIA")
    error_msg = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
    logger.error(error_msg, exc_info=True)
    console.print(f"[red]❌ {error_msg}[/red]")
    return error_msg

def validate_input(data: Any, expected_type: type, field_name: str = "input") -> bool:
    """
    Validate input data type
    
    Args:
        data: Data to validate
        expected_type: Expected data type
        field_name: Name of the field being validated
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, expected_type):
        error_msg = f"Invalid {field_name}: expected {expected_type.__name__}, got {type(data).__name__}"
        console.print(f"[yellow]⚠️ {error_msg}[/yellow]")
        return False
    return True

def format_response(response: Any, status: str = "success") -> Dict[str, Any]:
    """
    Format response in a consistent structure
    
    Args:
        response: The response data
        status: Status of the response (success, error, warning)
    
    Returns:
        Formatted response dictionary
    """
    return {
        "status": status,
        "data": response,
        "timestamp": "2025-06-18T00:00:00Z"  # In a real app, use datetime.now()
    }

def print_startup_banner():
    """Print startup banner for MRIA"""
    console.print("""
[bold blue]
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🏥 MRIA - Medical Retrivel Intelligence Assistant 🏥      ║
║                                                              ║
║    Empowering Healthcare Professionals with AI              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
[/bold blue]
    """)
    console.print("[green]✅ System initialized successfully![/green]\n")

def print_shutdown_message():
    """Print shutdown message"""
    console.print("\n[blue]👋 Thank you for using MRIA! Stay safe and keep healing! 🏥[/blue]")