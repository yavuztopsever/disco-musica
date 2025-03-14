"""
Main module for Disco Musica application.

This module launches the Disco Musica application.
"""

import os
import sys
from pathlib import Path

from modules.ui import DiscoMusicaUI


def main():
    """
    Main function to launch the Disco Musica application.
    """
    print("Starting Disco Musica - AI Music Generation Application")
    
    # Create data and model directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)
    os.makedirs("models/pretrained", exist_ok=True)
    os.makedirs("models/finetuned", exist_ok=True)
    
    # Launch the UI
    ui = DiscoMusicaUI()
    ui.launch_interface()


if __name__ == "__main__":
    main()