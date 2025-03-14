"""
Main module for Disco Musica application.

This module launches the Disco Musica application.
"""

import os
import sys
import argparse
from pathlib import Path

from modules.core.config import config, initialize_config
from modules.interfaces.gradio_ui import launch_gradio_interface


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Disco Musica - AI Music Generation Platform")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        help="Data directory"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory"
    )
    
    parser.add_argument(
        "--model-dir", 
        type=str, 
        help="Model directory"
    )
    
    parser.add_argument(
        "--default-model", 
        type=str, 
        help="Default text-to-music model"
    )
    
    parser.add_argument(
        "--sample-rate", 
        type=int, 
        help="Audio sample rate"
    )
    
    parser.add_argument(
        "--ui", 
        type=str, 
        choices=["gradio", "streamlit", "cli"],
        default="gradio",
        help="UI framework to use"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """
    Set up the application environment.
    
    Args:
        args: Command line arguments.
    """
    # Initialize config
    if args.config:
        initialize_config(args.config)
    
    # Set config values from command line args
    if args.data_dir:
        config.set("paths", "data_dir", args.data_dir)
    
    if args.output_dir:
        config.set("paths", "output_dir", args.output_dir)
    
    if args.model_dir:
        config.set("models", "model_cache_dir", args.model_dir)
    
    if args.default_model:
        config.set("models", "default_text_to_music", args.default_model)
    
    if args.sample_rate:
        config.set("audio", "sample_rate", args.sample_rate)
    
    # Create directories
    data_dir = Path(config.get("paths", "data_dir", "data"))
    output_dir = Path(config.get("paths", "output_dir", "outputs"))
    model_dir = Path(config.get("models", "model_cache_dir", "models"))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_dir / "raw", exist_ok=True)
    os.makedirs(data_dir / "processed", exist_ok=True)
    os.makedirs(data_dir / "datasets", exist_ok=True)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "audio", exist_ok=True)
    os.makedirs(output_dir / "midi", exist_ok=True)
    os.makedirs(output_dir / "visualizations", exist_ok=True)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir / "pretrained", exist_ok=True)
    os.makedirs(model_dir / "finetuned", exist_ok=True)
    
    # Set up logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    else:
        import logging
        logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function to launch the Disco Musica application.
    """
    print("Starting Disco Musica - AI Music Generation Platform")
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args)
    
    # Launch UI
    if args.ui == "gradio":
        launch_gradio_interface()
    elif args.ui == "streamlit":
        print("Streamlit UI not yet implemented")
    elif args.ui == "cli":
        print("CLI not yet implemented")
    else:
        print(f"Unknown UI framework: {args.ui}")


if __name__ == "__main__":
    main()