#!/usr/bin/env python3
"""
Disco Musica - AI Music Generation Application

This is the main entry point for the application.
"""

import os
import argparse
import logging
from pathlib import Path

# Configure logging
from utils.logging_utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from modules.interfaces.gradio_ui import launch_ui
from modules.services.generation_service import get_generation_service
from modules.services.model_service import get_model_service
from modules.services.output_service import get_output_service


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Disco Musica - AI Music Generation")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="ui", 
        choices=["ui", "cli"],
        help="Application mode (ui or cli)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to store outputs"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize services
    model_service = get_model_service()
    generation_service = get_generation_service()
    output_service = get_output_service()
    
    # Set output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    
    # Launch UI or CLI mode
    if args.mode == "ui":
        logger.info("Starting Disco Musica in UI mode")
        launch_ui(debug=args.debug)
    else:
        logger.info("Starting Disco Musica in CLI mode")
        # CLI mode would be implemented here
        print("CLI mode not yet implemented")
    

if __name__ == "__main__":
    main()