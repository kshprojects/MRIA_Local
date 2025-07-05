# Simple interface to run the system
# Just imports and calls agent.run_query()

#!/usr/bin/env python3
"""
MRIA - Medical Research Intelligence Assistant
Main entry point for the application
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.agent import run_query
from src.utils import setup_logging, print_startup_banner, print_shutdown_message, handle_error

def main():
    """Main entry point for MRIA application"""
    
    # Setup logging
    logger = setup_logging("INFO")
    
    try:
        # Print startup banner
        print_startup_banner()
        
        # Set event loop policy for Windows compatibility
        if os.name == "nt":  # For Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the main query loop
        asyncio.run(run_query())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print_shutdown_message()
    except Exception as e:
        handle_error(e, "main application")
        sys.exit(1)
    finally:
        print_shutdown_message()

if __name__ == "__main__":
    main()